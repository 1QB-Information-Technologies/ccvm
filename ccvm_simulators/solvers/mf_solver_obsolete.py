from ccvm_simulators.solvers.ccvm_solver import CCVMSolver
from ccvm_simulators.solution import Solution
from ccvm_simulators.post_processor.factory import PostProcessorFactory
import torch
import numpy as np
import torch.distributions as tdist
import time

MF_SCALING_MULTIPLIER = 0.1
"""The value used by the MFSolver when calculating a scaling value in
super.get_scaling_factor()"""


class MFSolver(CCVMSolver):
    """Constructor method"""

    def __init__(
        self,
        device,
        problem_category="boxqp",
        batch_size=1000,
    ):
        """
        Args:
            device (str): The device to use for the solver. Can be "cpu" or "cuda".
            problem_category (str): The category of problem to solve. Can be one
                of "boxqp". Defaults to "boxqp".
            batch_size (int): The number of times to solve a problem instance
                simultaneously. Defaults to 1000.

        Raises:
            ValueError: If the problem category is not supported by the solver.

        Returns:
            DLSolver: The DLSolver object.
        """
        super().__init__(device)
        self.batch_size = batch_size
        self._scaling_multiplier = MF_SCALING_MULTIPLIER
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
                * pump (float)
                * feedback_scale (float)
                * j (float)
                    * The measurement strength
                * S (float or vector of float with size 'problem_size')
                    * The enforced saturation value
                * lr (float)
                    * The learning rate
                * noise_ratio (float)

            With values, the parameter key might look like this::

                {
                    20:
                        {
                            "pump": 2.5,
                            "feedback_scale": 400,
                            "j": 399,
                            "S": 20.0,
                            "lr": 0.0025,
                            "iterations": 15000
                        },
                    30:
                        {
                            "pump": 3.0,
                            "feedback_scale": 250,
                            "j": 399,
                            "S": 20.0,
                            "lr": 0.0025,
                            "iterations": 15000
                        },
                }

        Raises:
            ValueError: If the parameter key does not contain the solver-specific
                combination of keys described above.
        """
        return self._parameter_key

    @parameter_key.setter
    def parameter_key(self, parameters):
        expected_mfparameter_key_set = set(
            ["pump", "feedback_scale", "j", "S", "lr", "iterations"]
        )
        parameter_key_list = parameters.values()
        # Iterate over the parameters for each given problem size
        for parameter_key in parameter_key_list:
            if parameter_key.keys() != expected_mfparameter_key_set:
                # The parameter key is not valid for this solver
                raise ValueError(
                    "The parameter key is not valid for this solver. Expected keys: "
                    + str(expected_mfparameter_key_set)
                    + " Given keys: "
                    + str(parameter_key.keys())
                )

        # If we get here, the parameter key is valid
        self._parameter_key = parameters
        self._is_tuned = False

    def _method_selector(self, problem_category):
        """Set methods relevant to this category of problem

        Args:
            problem_category (str): The category of problem to solve. Can be one
            of "boxqp".

        Raises:
            ValueError: If the problem category is not supported by the solver.
        """
        if problem_category.lower() == "boxqp":
            self.calculate_grads = self._calculate_grads_boxqp
            self.change_variables = self._change_variables_boxqp
            self.fit_to_constraints = self._fit_to_constraints_boxqp
        else:
            raise ValueError(
                f"The given problem category is not valid. Given category:"
                f" {problem_category}"
            )

    def _calculate_grads_boxqp(
        self,
        mu,
        mu_tilde,
        sigma,
        q_matrix,
        v_vector,
        pump,
        wiener_increment,
        j,
        g,
        S,
        fs,
    ):
        """We treat the SDE that simulates the CIM of NTT as gradient
        calculation. Original SDE considers only quadratic part of the objective
        function. Therefore, we need to modify and add linear part of the QP to
        the SDE.

        Args:
            mu (torch.Tensor): Mean-field amplitudes
            mu_tilde (torch.Tensor): Mean-field measured amplitudes
            sigma (torch.Tensor): Variance of the in-phase position operator
            q_matrix (torch.tensor): The Q matrix describing the BoxQP problem.
            v_vector (torch.tensor): The V vector describing the BoxQP problem.
            pump (float): Instantaneous pump value
            wiener_increment (torch.Tensor): The Wiener process
            j (float): The measurement strength
            g (float): The nonlinearity coefficient
            S (float): The enforced saturation value
            fs (float): The coefficient of the feedback term.

        Returns:
            tuple: The gradients of the mean-field amplitudes and the variance.
        """
        mu_pow = torch.pow(mu, 2)

        mu_term1 = (-(1 + j) + pump - g**2 * mu_pow) * mu
        mu_term2_1 = (
            -(1 / 4) * (torch.einsum("bi,ij -> bj", mu_tilde / S + 1, q_matrix)) / S
        )
        mu_term2_2 = -v_vector / S / 2
        mu_term3 = np.sqrt(j) * (sigma - 0.5) * wiener_increment

        sigma_term1 = 2 * (-(1 + j) + pump - 3 * g**2 * mu_pow) * sigma
        sigma_term2 = -2 * j * (sigma - 0.5).pow(2)
        sigma_term3 = (1 + j) + 2 * g**2 * mu_pow

        grads_mu = mu_term1 + fs * (mu_term2_1 + mu_term2_2) + mu_term3
        grads_sigma = sigma_term1 + sigma_term2 + sigma_term3

        return grads_mu, grads_sigma

    def _change_variables_boxqp(self, problem_variables, S=1):
        """Perform a change of variables to enforce the box constraints.

        Args:
            problem_variables (torch.Tensor): The variables to change.
            S (float or torch.tensor): The enforced saturation value. Defaults to 1

        Returns:
            torch.Tensor: The changed variables.
        """
        return 0.5 * (problem_variables / S + 1)

    def _fit_to_constraints_boxqp(self, mu_tilde, lower_clamp, upper_clamp):
        """Clamps the values of mu_tilde to be within the box constraints

        Args:
            mu_tilde (torch.Tensor): The mean-field measured amplitudes.
            lower_clamp (float or torch.tensor): The lower bound of the box constraints.
            upper_clamp (float or torch.tensor): The upper bound of the box constraints.

        Returns:
            torch.Tensor: The clamped values of mu_tilde, now within the box constraints.
        """
        mu_tilde_clamped = torch.clamp(mu_tilde, lower_clamp, upper_clamp)
        return mu_tilde_clamped

    def _append_samples_to_file(self, mu_sample, sigma_sample, evolution_file_object):
        """Saves samples of the mean-field amplitudes and the variance of the in-phase
        position operator to a file.
        The end file will contain the values of the mu_sample followed by the sigma_sample.
        Each line corresponds to a row in the tensor, with tab-delineated values.

        Args:
            mu_sample (torch.Tensor): The sample of mean-field amplitudes to add to the file.
            Expected Dimensions: problem_size x num_samples.
            sigma_sample (torch.Tensor): The sample of the variance of the in-phase position
            operator to add to the file. Expected Dimensions: problem_size x num_samples.
            evolution_file_object (io.TextIOWrapper): The file object of the file to save
            the samples to.
        """
        # Save the mu samples to the file
        mu_rows = mu_sample.shape[0]  # problem_size
        mu_columns = mu_sample.shape[1]  # num_samples
        for nn in range(mu_rows):
            for ii in range(mu_columns):
                evolution_file_object.write(str(round(mu_sample[nn, ii].item(), 4)))
                if ii != mu_columns - 1:
                    evolution_file_object.write("\t")
            evolution_file_object.write("\n")

        # Save the sigma samples to the file
        sigma_rows = sigma_sample.shape[0]  # problem_size
        sigma_columns = sigma_sample.shape[1]  # num_samples
        for nn in range(sigma_rows):
            for ii in range(sigma_columns):
                evolution_file_object.write(str(round(sigma_sample[nn, ii].item(), 4)))
                if ii != sigma_columns - 1:
                    evolution_file_object.write("\t")
            evolution_file_object.write("\n")

    def tune(self, instances, post_processor, g=0.01):
        """Determines the best parameters for the solver to use by adjusting each
        parameter over a number of iterations on the problems in the given set of
        problems instances. The `parameter_key` attribute of the solver will be
        updated with the best parameters found.

        Args:
            instances (list): A list of problem instances to tune the solver on.
            post_processor (str): The name of the post processor to use to process the
                results of the solver. None if no post processing is desired.
                Defaults to None.
            g (float): The nonlinearity coefficient. Defaults to 0.01.
        """
        # TODO: This implementation is a placeholder; full implementation is a
        #       future consideration
        self.is_tuned = True

    def solve(
        self,
        instance,
        post_processor=None,
        g=0.01,
        pump_rate_flag=True,
        evolution_step_size=None,
        evolution_file=None,
    ):
        """Solves the given problem instance using the tuned or specified
        parameters in the parameter key.

        Args:
            instance (boxqp.boxqp.ProblemInstance): The problem to solve.
            post_processor (str): The name of the post processor to use to process
                the results of the solver. None if no post processing is desired.
            g (float, optional): The nonlinearity coefficient. Defaults to 0.01
            pump_rate_flag (bool, optional): Whether or not to scale the pump rate
                based on the iteration number. If False, the pump rate will be 1.0.
                Defaults to True.
            evolution_step_size (int): If set, the mu/sigma values will be sampled once per number of
                iterations equivalent to the value of this variable. At the end of the solve process,
                the best batch of sampled values will be written to a file that can be specified by
                setting the evolution_file parameter.Defaults to None, meaning no problem variables
                will be written to the file.
            evolution_file (str): The file to save the best set of mu/sigma samples to. Only revelant
                when evolution_step_size is set. If a file already exists with the same name,
                it will be overwritten. Defaults to None, which generates a filename based on
                the problem instance name.

        Returns:
            solution (Solution): The solution to the problem instance.
        """
        # If the instance and the solver don't specify the same device type, raise an
        # error
        if instance.device != self.device:
            raise ValueError(
                f"The device type of the instance ({instance.device}) and the solver"
                f" ({self.device}) must match."
            )

        # Get problem from problem instance
        problem_size = instance.problem_size
        q_matrix = instance.q_matrix
        v_vector = instance.v_vector

        # Get solver setup variables
        batch_size = self.batch_size
        device = self.device

        # Get parameters from parameter_key
        try:
            pump = self.parameter_key[problem_size]["pump"]
            lr = self.parameter_key[problem_size]["lr"]
            iterations = self.parameter_key[problem_size]["iterations"]
            j = self.parameter_key[problem_size]["j"]
            feedback_scale = self.parameter_key[problem_size]["feedback_scale"]
            S = self.parameter_key[problem_size]["S"]

            # If S is a 1-D tensor, convert it to to a 2-D tensor
            if torch.is_tensor(S) and S.ndim == 1:
                # Dimension indexing in pytorch starts at 0
                if S.size(dim=0) == problem_size:
                    S = torch.outer(torch.ones(batch_size), S)
                else:
                    raise ValueError("Tensor S size should be equal to problem size.")
        except KeyError as e:
            raise KeyError(
                f"The parameter '{e.args[0]}' for the given instance size is not"
                " defined."
            ) from e

        # Start timing the solve process
        solve_time_start = time.time()

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
            mu_sample = torch.zeros(
                (batch_size, problem_size, num_samples), device="cpu"
            )
            sigma_sample = torch.zeros(
                (batch_size, problem_size, num_samples), device="cpu"
            )
            samples_taken = 0

        # Initialize tensor variables on the device that will be used to perform
        # the calculations

        mu = torch.zeros((batch_size, problem_size), dtype=torch.float, device=device)
        sigma = torch.ones(
            (batch_size, problem_size), dtype=torch.float, device=device
        ) * (1 / 2)

        wiener_dist = tdist.Normal(
            torch.tensor([0.0] * batch_size, device=device),
            torch.tensor([1.0] * batch_size, device=device),
        )

        # Perform the solve over the specified number of iterations
        pump_rate = 1
        for i in range(iterations):

            j_i = j * np.exp(-(i + 1) / iterations * 3.0)
            wiener = wiener_dist.sample((problem_size,)).transpose(0, 1)
            wiener_increment = wiener / np.sqrt(lr)
            mu_tilde = mu + np.sqrt(1 / (4 * j_i)) * wiener_increment
            mu_tilde_c = self.fit_to_constraints(mu_tilde, -S, S)

            if pump_rate_flag:
                pump_rate = (i + 1) / iterations

            instantaneous_pump = pump * pump_rate + 1 + j_i

            (grads_mu, grads_sigma) = self.calculate_grads(
                mu,
                mu_tilde_c,
                sigma,
                q_matrix,
                v_vector,
                instantaneous_pump,
                wiener_increment,
                j_i,
                g,
                S,
                feedback_scale,
            )
            mu += lr * grads_mu
            sigma += lr * grads_sigma

            # If evolution_step_size is specified, save the values if this iteration
            # aligns with the step size or if this is the last iteration
            if evolution_step_size and (
                i % evolution_step_size == 0 or i + 1 >= iterations
            ):
                # Update the record of the sample values with the values found at
                # this iteration
                mu_sample[:, :, samples_taken] = mu
                sigma_sample[:, :, samples_taken] = sigma
                samples_taken += 1

        mu_tilde = self.fit_to_constraints(mu_tilde, -S, S)

        solve_time = time.time() - solve_time_start

        # Run the post processor on the results, if specified
        if post_processor:
            post_processor_object = PostProcessorFactory.create_postprocessor(
                post_processor
            )

            problem_variables = post_processor_object.postprocess(
                self.change_variables(mu_tilde, S), q_matrix, v_vector, device=device
            )
            pp_time = post_processor_object.pp_time
        else:
            problem_variables = self.change_variables(mu_tilde, S)
            pp_time = 0.0

        objval = instance.compute_energy(problem_variables)

        if evolution_step_size:
            # Write samples to file
            # Overwrite file if it exists
            open(evolution_file, "w")

            # Get the indices of the best objective values over the sampled iterations
            # to use to get and save the best sampled values of mu and sigma
            batch_index = torch.argmax(-objval)
            with open(evolution_file, "a") as evolution_file_obj:
                self._append_samples_to_file(
                    mu_sample=mu_sample[batch_index],
                    sigma_sample=sigma_sample[batch_index],
                    evolution_file_object=evolution_file_obj,
                )

        solution = Solution(
            problem_size=problem_size,
            batch_size=batch_size,
            instance_name=instance.name,
            iterations=iterations,
            objective_values=objval,
            solve_time=solve_time,
            pp_time=pp_time,
            optimal_value=instance.optimal_sol,
            variables={
                "problem_variables": problem_variables,
                "mu": mu,
                "sigma": sigma,
            },
            device=device,
        )

        # Add evolution filename to solution if it was generated
        if evolution_step_size:
            solution.evolution_file = evolution_file

        return solution
