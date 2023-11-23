from ccvm_simulators.solvers import CCVMSolver
from ccvm_simulators.solvers.algorithms import AdamParameters
from ccvm_simulators.solution import Solution
from ccvm_simulators.post_processor.factory import PostProcessorFactory
import torch
import numpy as np
import torch.distributions as tdist
import time

LANGEVIN_SCALING_MULTIPLIER = 0.5
"""The value used by the LangevinSolver when calculating a scaling value in
super.get_scaling_factor()"""


class LangevinSolver(CCVMSolver):
    """The LangevinSolver class models typical Langevin dynamics as a system of
    SDE."""

    def __init__(
        self,
        device,
        problem_category="boxqp",
        batch_size=1000,
        S=1,
    ):
        """
        Args:
            device (str): The device to use for the solver. Can be "cpu" or "cuda".
            problem_category (str): The category of problem to solve. Can be one of
                "boxqp". Defaults to "boxqp".
            batch_size (int): The number of times to solve a problem instance
                simultaneously. Defaults to 1000.
            S (float or torch.tensor): Enforced saturation value. Defaults to 1.

        Raises:
            ValueError: If the problem category is not supported by the solver.

        Returns:
            LangevinSolver: The LangevinSolver object.
        """
        super().__init__(device)
        self.batch_size = batch_size
        self.S = S
        self._scaling_multiplier = LANGEVIN_SCALING_MULTIPLIER
        # Use the method selector to choose the problem-specific methods to use
        self._method_selector(problem_category)

    @property
    def parameter_key(self):
        """The parameters that will be used by the solver when solving the problem.

        Note:
            Setting this parameter after calling tune() will overwrite tuned parameters.

        The parameter_key must match the following format:

            * key: problem size (the number of variables in the problem).
            * value: dict with these keys:
                * pump (float),
                * dt (float),
                * iterations (int),
                * sigma (float),
                * feedback_scale (float)

            With values, the parameter key might look like this::

                {
                    20: {"dt": 0.005, "iterations": 15000, "sigma":0.02, "feedback_scale": 1.0}
                }

        Raises:
            ValueError: If the parameter key does not contain the solver-specific
                combination of keys described above.
        """
        return self._parameter_key

    @parameter_key.setter
    def parameter_key(self, parameters):
        expected_dlparameter_key_set = set(
            ["dt", "iterations", "sigma", "feedback_scale"]
        )
        parameter_key_list = parameters.values()
        # Iterate over the parameters for each given problem size
        for parameter_key in parameter_key_list:
            if parameter_key.keys() != expected_dlparameter_key_set:
                # The parameter key is not valid for this solver
                raise ValueError(
                    "The parameter key is not valid for this solver. Expected keys: "
                    + str(expected_dlparameter_key_set)
                    + " Given keys: "
                    + str(parameter_key.keys())
                )

        # If we get here, the parameter key is valid
        self._parameter_key = parameters
        self._is_tuned = False

    def _calculate_drift_boxqp(self, c, S=1):
        """We treat the SDE that simulates the CIM of NTT as drift
        calculation.

        Args:
            c (torch.Tensor): In-phase amplitudes of the solver
            S (float): The saturation value of the amplitudes. Defaults to 1.

        Returns:
            tensor: The calculated change in the variable amplitude.
        """

        c_drift_1 = torch.einsum("bi,ij -> bj", c, self.q_matrix)
        c_drift_2 = self.v_vector

        c_drift = -c_drift_1 - c_drift_2

        return c_drift

    def _calculate_grads_boxqp(self, c):
        """We treat the SDE that simulates the CIM of NTT as gradient
        calculation. Original SDE considers only quadratic part of the objective
        function. Therefore, we need to modify and add linear part of the QP to
        the SDE.

        Args:
            c (torch.Tensor): In-phase amplitudes of the solver

        Returns:
            tensor: The calculated change in the variable amplitude.
        """

        c_grad_1 = torch.einsum("bi,ij -> bj", c, self.q_matrix)
        c_grad_2 = self.v_vector

        c_grads = -c_grad_1 - c_grad_2

        return c_grads

    def _change_variables_boxqp(self, problem_variables, S=1):
        """Perform a change of variables to enforce the box constraints.

        Args:
            problem_variables (torch.Tensor): The variables to change.
            S (float): The saturation value of the amplitudes. Defaults to 1.

        Returns:
            torch.Tensor: The changed variables.
        """
        return 0.5 * (problem_variables / S + 1)

    def _fit_to_constraints_boxqp(self, c, lower_clamp, upper_clamp):
        """Clamps the values of c to be within the box constraints

        Args:
            c (torch.Tensor): The variables to clamp.
            lower_clamp (float or torch.tensor): The lower bound of the box constraints.
            upper_clamp (float or torch.tensor): The upper bound of the box constraints.

        Returns:
            torch.Tensor: The clamped variables.
        """

        c_clamped = torch.clamp(c, lower_clamp, upper_clamp)
        return c_clamped

    def _append_samples_to_file(self, c_sample, s_sample, evolution_file_object):
        """Saves samples of the amplitude values to a file.
        The end file will contain the values of the c_sample followed by the s_sample.
        Each line corresponds to a row in the tensor, with tab-delineated values.

        Args:
            c_sample (torch.Tensor): The sample of in-phase amplitudes to add to the
            file. Expected Dimensions: problem_size x num_samples
            s_sample (torch.Tensor): The sample of quadrature amplitudes to add to the
            file. Expected Dimensions: problem_size x num_samples
            evolution_file_object (io.TextIOWrapper): The file object of the file to save
            the samples to.
        """
        # Save the c samples to the file
        c_rows = c_sample.shape[0]  # problem_size
        c_columns = c_sample.shape[1]  # num_samples
        for nn in range(c_rows):
            for ii in range(c_columns):
                evolution_file_object.write(str(round(c_sample[nn, ii].item(), 4)))
                evolution_file_object.write("\t")
            evolution_file_object.write("\n")

        # Save the s samples to the file
        s_rows = s_sample.shape[0]  # problem_size
        s_columns = s_sample.shape[1]  # num_samples
        for nn in range(s_rows):
            for ii in range(s_columns):
                evolution_file_object.write(str(round(s_sample[nn, ii].item(), 4)))
                evolution_file_object.write("\t")
            evolution_file_object.write("\n")

    def tune(self, instances, post_processor=None, pump_rate_flag=True, g=0.05):
        """Determines the best parameters for the solver to use by adjusting each
        parameter over a number of iterations on the problems in the given set of
        problems instances. The `parameter_key` attribute of the solver will be
        updated with the best parameters found.

        Args:
            instances (list): A list of problem instances to tune the solver on.
            post_processor (str): The name of the post processor to use to process the
                results of the solver. None if no post processing is desired.
                Defaults to None.
            pump_rate_flag (bool): Whether or not to scale the pump rate based on the
            iteration number. If False, the pump rate will be 1.0. Defaults to True.
            g (float): The nonlinearity coefficient. Defaults to 0.05.
        """
        # TODO: This implementation is a placeholder; full implementation is a
        #       future consideration
        self.is_tuned = True

    def _solve(
        self,
        problem_size,
        batch_size,
        device,
        S,
        dt,
        iterations,
        sigma,
        feedback_scale,
        evolution_step_size,
        samples_taken,
    ):
        """Solves the given problem instance using the original Langevin solver.

        Args:
            problem_size (int): instance size.
            batch_size (int): The number of times to solve a problem instance
            device (str): The device to use for the solver. Can be "cpu" or "cuda".
            S (float): Saturation bound.
            dt (float): Simulation time step.
            iterations (int): number of steps.
            sigma (float): diffusion constant.
            feedback_scale (float): feedback scale.
            evolution_step_size (int): If set, the c/s values will be sampled once
                per number of iterations equivalent to the value of this variable.
                At the end of the solve process, the best batch of sampled values
                will be written to a file that can be specified by setting the evolution_file parameter.
            samples_taken (int): sample slice.

        Returns:
            c (tensor): random variable.
        """
        # Initialize tensor variables on the device that will be used to perform the
        # calculations
        c = torch.zeros((batch_size, problem_size), dtype=torch.float, device=device)
        wiener_dist_c = tdist.Normal(
            torch.tensor([0.0] * batch_size, device=device),
            torch.tensor([1.0] * batch_size, device=device),
        )

        # Perform the solve over the specified number of iterations
        for i in range(iterations):
            c_drift = self.calculate_drift(c, S)

            wiener_increment_c = wiener_dist_c.sample((problem_size,)).transpose(
                0, 1
            ) * np.sqrt(dt)

            c += dt * feedback_scale * c_drift + sigma * wiener_increment_c
            # Ensure variables are within any problem constraints
            # The lower bound is determined by ell=0, and upper bound by u=1
            c = self.fit_to_constraints(c, 0, 1.0)

            # If evolution_step_size is specified, save the values if this iteration
            # aligns with the step size or if this is the last iteration
            if evolution_step_size and (
                i % evolution_step_size == 0 or i + 1 >= iterations
            ):
                # Update the record of the sample values with the values found at
                # this iteration
                self.c_sample[:, :, samples_taken] = c
                samples_taken += 1
        
        return c

    def _solve_adam(
        self,
        problem_size,
        batch_size,
        device,
        S,
        dt,
        iterations,
        sigma,
        feedback_scale,
        evolution_step_size,
        samples_taken,
        hyperparameters,
    ):
        """Solves the given problem instance using the Langevin solver with Adam algorithm.

        Args:
            problem_size (int): instance size.
            batch_size (int): The number of times to solve a problem instance
            device (str): The device to use for the solver. Can be "cpu" or "cuda".
            S (float): Saturation bound.
            dt (float): Simulation time step.
            iterations (int): number of steps.
            sigma (float): diffusion constant.
            feedback_scale (float): feedback scale.
            evolution_step_size (int): If set, the c/s values will be sampled once
                per number of iterations equivalent to the value of this variable.
                At the end of the solve process, the best batch of sampled values
                will be written to a file that can be specified by setting the evolution_file parameter.
            samples_taken (int): sample slice.
            hyperparameters (dict): Hyperparameters for Adam algorithm.

        Returns:
            c (tensor): random variable.
        """
        # Initialize tensor variables on the device that will be used to perform the
        # calculations
        c = torch.zeros((batch_size, problem_size), dtype=torch.float, device=device)
        wiener_dist_c = tdist.Normal(
            torch.tensor([0.0] * batch_size, device=device),
            torch.tensor([1.0] * batch_size, device=device),
        )

        alpha = hyperparameters["alpha"]
        beta1 = hyperparameters["beta1"]
        beta2 = hyperparameters["beta2"]
        epsilon = 1e-8

        # Choose desired update method.
        if hyperparameters["add_assign"]:
            update_grads_with_moment2 = (
                lambda grads, mhat, vhat: grads
                + alpha * torch.div(mhat, torch.sqrt(vhat) + epsilon)
            )

            update_grads_without_moment2 = lambda grads, mhat: grads + alpha * mhat_c
        else:
            update_grads_with_moment2 = lambda grads, mhat, vhat: alpha * torch.div(
                mhat, torch.sqrt(vhat) + epsilon
            )

            update_grads_without_moment2 = lambda grads, mhat: alpha * mhat_c

        # Initialize first moment vector
        m_c = torch.zeros((batch_size, problem_size), dtype=torch.float, device=device)
        # Initialize second moment conditionally
        if not beta2 == 1.0:
            v_c = torch.zeros(
                (batch_size, problem_size), dtype=torch.float, device=device
            )
        else:
            v_c = None

        # Perform the solve with Adam over the specified number of iterations
        for i in range(iterations):
            # Calculate gradient
            c_grads = self.calculate_grads(c)

            # Update biased first moment estimate
            m_c = beta1 * m_c + (1.0 - beta1) * c_grads

            # Compute bias correction in 1st moment
            beta1i = 1.0 - beta1 ** (i + 1)
            mhat_c = m_c / beta1i

            # Conditional second moment estimation
            if not beta2 == 1.0:
                # Update biased 2nd moment estimate
                v_c = beta2 * v_c + (1.0 - beta2) * torch.pow(c_grads, 2)

                # Compute bias correction in 2nd moment
                beta2i = 1.0 - beta2 ** (i + 1)
                vhat_c = v_c / beta2i

                # Compute bias corrected grad using 1st and 2nd moments
                # in the form of element-wise division
                c_grads = update_grads_with_moment2(c_grads, mhat_c, vhat_c)
            else:
                # Compute bias corrected grad only with 1st moment
                c_grads = update_grads_without_moment2(c_grads, mhat_c)

            wiener_increment_c = wiener_dist_c.sample((problem_size,)).transpose(
                0, 1
            ) * np.sqrt(dt)

            c += dt * feedback_scale * c_grads + sigma * wiener_increment_c
            # Ensure variables are within any problem constraints
            # The lower bound is determined by ell=0, and upper bound by u=1
            c = self.fit_to_constraints(c, 0, 1.0)

            # If evolution_step_size is specified, save the values if this iteration
            # aligns with the step size or if this is the last iteration
            if evolution_step_size and (
                i % evolution_step_size == 0 or i + 1 >= iterations
            ):
                # Update the record of the sample values with the values found at
                # this iteration
                self.c_sample[:, :, samples_taken] = c
                samples_taken += 1
        
        return c

    def __call__(
        self,
        instance,
        post_processor=None,
        evolution_step_size=None,
        evolution_file=None,
        algorithm_parameters=None,
    ):
        """Solves the given problem instance by choosing one of the available Langevin solvers.

        Args:
            instance (ProblemInstance): The problem instance to solve.
            post_processor (str): The name of the post processor to use to process the results of the solver.
                None if no post processing is desired. Defaults to None.
            evolution_step_size (int): If set, the c/s values will be sampled once
                per number of iterations equivalent to the value of this variable.
                At the end of the solve process, the best batch of sampled values
                will be written to a file that can be specified by setting the evolution_file parameter.
                Defaults to None, meaning no problem variables will be written to the file.
            evolution_file (str): The file to save the best set of c/s samples to.
                Only revelant when evolution_step_size is set.
                If a file already exists with the same name, it will be overwritten.
                Defaults to None, which generates a filename based on the problem instance name.
            algorithm_parameters (None, AdamParameters): Specify for the solver to use a specialized algorithm by passing in
                an instance of the algorithm's parameters class. Options include: AdamParameters.
                Defaults to None, which uses the original Langevin solver.

        Returns:
            solution (Solution): The solution to the problem instance.
        """
        # If the instance and the solver don't specify the same device type, raise
        # an error
        if instance.device != self.device:
            raise ValueError(
                f"The device type of the instance ({instance.device}) and the solver"
                f" ({self.device}) must match."
            )

        # Get problem from problem instance
        problem_size = instance.problem_size
        self.q_matrix = instance.q_matrix
        self.v_vector = instance.v_vector

        # Get solver setup variables
        S = self.S
        batch_size = self.batch_size
        device = self.device

        # Get parameters from parameter_key
        try:
            dt = self.parameter_key[problem_size]["dt"]
            iterations = self.parameter_key[problem_size]["iterations"]
            sigma = self.parameter_key[problem_size]["sigma"]
            feedback_scale = self.parameter_key[problem_size]["feedback_scale"]

        except KeyError as e:
            raise KeyError(
                f"The parameter '{e.args[0]}' for the given instance size is not defined."
            ) from e

        # If S is a 1-D tensor, convert it to to a 2-D tensor
        if torch.is_tensor(S) and S.ndim == 1:
            # Dimension indexing in pytorch starts at 0
            if S.size(dim=0) == problem_size:
                S = torch.outer(torch.ones(batch_size), S)
            else:
                raise ValueError("Tensor S size should be equal to problem size.")

        # Start the timer for the solve
        solve_time_start = time.time()

        samples_taken = None
        self.c_sample = None  # variable for random sample
        if evolution_step_size:
            # Check that the value is valid
            if evolution_step_size < 1:
                raise ValueError(
                    f"The evolution step size must be greater than or equal to 1."
                )
            # Generate evolution file name
            if evolution_file is None:
                evolution_file = f"./{instance.name}_evolution.txt"

            # Get the number of samples to save
            # Find the number of full steps that will be taken
            num_steps = int(iterations / evolution_step_size)
            # We will also capture the first iteration through
            num_samples = num_steps + 1
            # And capture the last iteration if the step size doesn't evenly divide
            if iterations % evolution_step_size != 0:
                num_samples += 1

            # Initialize tensors
            # Store on CPU to keep the memory usage lower on the GPU
            self.c_sample = torch.zeros(
                (batch_size, problem_size, num_samples),
                dtype=torch.float,
                device="cpu",
            )
            samples_taken = 0

        if algorithm_parameters is None:
            # Use the original Langevin solver
            c = self._solve(
                problem_size,
                batch_size,
                device,
                S,
                dt,
                iterations,
                sigma,
                feedback_scale,
                evolution_step_size,
                samples_taken,
            )
        elif isinstance(algorithm_parameters, AdamParameters):
            # Use the Langevin solver with the Adam algorithm
            c = self._solve_adam(
                problem_size,
                batch_size,
                device,
                S,
                dt,
                iterations,
                sigma,
                feedback_scale,
                evolution_step_size,
                samples_taken,
                algorithm_parameters.to_dict(),
            )
        else:
            raise ValueError(
                f"Solver option type {type(algorithm_parameters)} is not supported."
            )

        # Stop the timer for the solve
        solve_time = time.time() - solve_time_start

        # Run the post processor on the results, if specified
        if post_processor:
            post_processor_object = PostProcessorFactory.create_postprocessor(
                post_processor
            )

            problem_variables = post_processor_object.postprocess(
                c, self.q_matrix, self.v_vector
            )
            pp_time = post_processor_object.pp_time
        else:
            problem_variables = c
            pp_time = 0.0

        # Calculate the objective value
        objval = instance.compute_energy(problem_variables)

        if evolution_step_size:
            # Write samples to file
            # Overwrite file if it exists
            open(evolution_file, "w")

            # Get the indices of the best objective values over the sampled iterations
            # to use to get and save the best sampled values of c and s
            batch_index = torch.argmax(-objval)
            with open(evolution_file, "a") as evolution_file_obj:
                self._append_samples_to_file(
                    c_sample=self.c_sample[batch_index],
                    evolution_file_object=evolution_file_obj,
                )

        solution = Solution(
            problem_size=instance.problem_size,
            batch_size=batch_size,
            instance_name=instance.name,
            iterations=iterations,
            objective_values=objval,
            solve_time=solve_time,
            pp_time=pp_time,
            optimal_value=instance.optimal_sol,
            best_value=instance.best_sol,
            num_frac_values=instance.num_frac_values,
            solution_vector=instance.solution_vector,
            variables={"problem_variables": problem_variables},
            device=self.device,
        )

        # Add evolution filename to solution if it was generated
        if evolution_step_size:
            solution.evolution_file = evolution_file

        return solution
