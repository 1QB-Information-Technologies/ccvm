from ccvm.solvers import CCVMSolver
from ccvm.solution import Solution
from ccvm.post_processor.factory import PostProcessorFactory
import torch
import numpy as np
import torch.distributions as tdist
import time

DL_SCALING_MULTIPLIER = 0.5
"""The value used by the DLSolver when calculating a scaling value in
super.get_scaling_factor()"""


class DLSolver(CCVMSolver):
    """The DLSolver class models the delay line coherent continuous-variable machine
    (DL-CCVM)."""

    def __init__(
        self,
        device,
        problem_category="boxqp",
        time_evolution_results=False,
        batch_size=1000,
    ):
        """
        Args:
            device (str): The device to use for the solver. Can be "cpu" or "cuda".
            problem_category (str): The category of problem to solve. Can be one of
            "boxqp". Defaults to "boxqp".
            time_evolution_results (bool): Whether to return the time evolution results
            for each iteration during the solve. Defaults to True.
            batch_size (int): The batch size of the problem. Defaults to 1000.

        Raises:
            ValueError: If the problem category is not supported by the solver.

        Returns:
            DLSolver: The DLSolver object.
        """
        super().__init__(device)
        self.time_evolution_results = time_evolution_results
        self.batch_size = batch_size
        self._scaling_multiplier = DL_SCALING_MULTIPLIER
        # Use the method selector to choose the problem-specific methods to use
        self._method_selector(problem_category)

    def _validate_parameters(self, parameters):
        """Validate the parameter key against the keys in the expected parameters for
        DLSolver.

        Args:
            parameters (dict): The parameters to validate. The parameters must match the
                format:
                {
                    <problem size>: <dict with set of keys:
                            p (pump),
                            lr (learning rate),
                            iter (iterations),
                            nr (noise_ratio)
                        >
                }
        For example:
                {
                    20: {"p": 2.0, "lr": 0.005, "iter": 15000, "nr": 10},
                    30: {"p": 2.0, "lr": 0.005, "iter": 15000, "nr": 5},
                }

        Raises:
            ValueError: If the parameter key is not valid for this solver.
        """
        expected_dlparameter_key_set = set(["p", "lr", "iter", "nr"])
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

    def _method_selector(self, problem_category):
        """Set methods relevant to this category of problem

        Args:
            problem_category (str): The category of problem to solve. Can be one of "boxqp".

        Raises:
            ValueError: If the problem category is not supported by the solver.
        """
        if problem_category.lower() == "boxqp":
            self.calculate_grads = self._calculate_grads_boxqp
            self.change_variables = self._change_variables_boxqp
            self.fit_to_constraints = self._fit_to_constraints_boxqp
        else:
            raise ValueError(
                "The given instance is not a valid problem category."
                f" Given category: {problem_category}"
            )

    def _calculate_grads_boxqp(self, c, s, q_matrix, c_vector, p, rate, S=1):
        """We treat the SDE that simulates the CIM of NTT as gradient
        calculation. Original SDE considers only quadratic part of the objective
        function. Therefore, we need to modify and add linear part of the QP to
        the SDE.

        Args:
            c (torch.Tensor): TODO
            s (torch.Tensor): TODO
            q_matrix (torch.Tensor): The coefficient matrix of the quadratic terms.
            c_vector (torch.Tensor): The coefficient vector of the linear terms.
            p (float): TODO
            rate (float): TODO
            S (float): TODO Defaults to 1.

        Returns:
            tuple: The gradients of the c and s variables.
        """

        c_pow = torch.pow(c, 2)
        s_pow = torch.pow(s, 2)

        if p > 1:
            S = np.sqrt(p - 1)

        c_grad_1 = 0.25 * torch.einsum("bi,ij -> bj", c / S + 1, q_matrix)
        c_grad_2 = torch.einsum("cj,cj -> cj", -1 + (p * rate) - c_pow - s_pow, c)
        c_grad_3 = c_vector / 2 / S

        s_grad_1 = 0.25 * torch.einsum("bi,ij -> bj", s / S + 1, q_matrix)
        s_grad_2 = torch.einsum("cj,cj -> cj", -1 - (p * rate) - c_pow - s_pow, s)
        s_grad_3 = c_vector / 2 / S

        c_grads = -c_grad_1 + c_grad_2 - c_grad_3
        s_grads = -s_grad_1 + s_grad_2 - s_grad_3
        return c_grads, s_grads

    def _change_variables_boxqp(self, problem_variables):
        """Perform a change of variables to enforce the box constraints.

        Args:
            problem_variables (torch.Tensor): The variables to change.

        Returns:
            torch.Tensor: The changed variables.
        """
        return 0.5 * (problem_variables + 1)

    def _fit_to_constraints_boxqp(self, c):
        """Clamps the values of c to be within the box constraints

        Args:
            c (torch.Tensor): The variables to clamp.

        Returns:
            torch.Tensor: The clamped variables.
        """
        c_clamped = torch.clamp(c, -1, 1)
        return c_clamped

    def tune(self, instances, post_processor=None, pump_rate_flag=True, g=0.05):
        """Determines the best parameters for the solver to use by adjusting each
        parameter over a number of iterations on the problems in the given set of
        problems instances. The `parameter_key` attribute of the solver will be
        updated with the best parameters found.

        Args:
            instances (list): A list of problem instances to tune the solver on.
            post_processor (PostProcessorType): The post processor to use to process
            the results of the solver. None if no post processing is desired.
            Defaults to None.
            pump_rate_flag (bool): Whether or not to scale the pump rate based on the
            iteration number. If False, the pump rate will be 1.0. Defaults to True.
            g (float): _description_ Defaults to 0.05.
        """
        # TODO: summary/descriptions
        # TODO: This implementation is a placeholder; full implementation is a
        #       future consideration
        self.is_tuned = True

    def solve(self, instance, post_processor=None, pump_rate_flag=True, g=0.05):
        """Solves the given problem instance using the DL-CCVM solver.

        Args:
            instance (ProblemInstance): The problem instance to solve.
            post_processor (PostProcessorType): The post processor to use to process
            the results of the solver. None if no post processing is desired.
            Defaults to None.
            pump_rate_flag (bool): Whether or not to scale the pump rate based on the
            iteration number. If False, the pump rate will be 1.0. Defaults to True.
            g (float): _description_ Defaults to 0.05.

        Returns:
            tuple: The solution to the problem instance and the timing values.
        """
        # If the instance and the solver don't specify the same device type, raise
        # an error
        if instance.device != self.device:
            raise ValueError(
                f"The device type of the instance ({instance.device}) and the solver"
                f" ({self.device}) must match."
            )

        # Get problem size from problem instance
        N = instance.N
        q_mat = instance.q
        c_vector = instance.c

        # Get solver setup variables
        batch_size = self.batch_size
        device = self.device
        time_evolution_results = self.time_evolution_results

        # Get parameters from parameter_key
        try:
            p = self.parameter_key[N]["p"]
            lr = self.parameter_key[N]["lr"]
            n_iter = self.parameter_key[N]["iter"]
            noise_ratio = self.parameter_key[N]["nr"]
        except KeyError as e:
            raise KeyError(
                f"The parameter '{e.args[0]}' for the given instance size is not defined."
            ) from e

        # Start the timer for the solve
        solve_time_start = time.time()

        # Initialize tensor variables on the device that will be used to perform the
        # calculations
        c = torch.zeros((batch_size, N), dtype=torch.float).to(device)
        s = torch.zeros((batch_size, N), dtype=torch.float).to(device)
        if time_evolution_results:
            c_time = torch.zeros((batch_size, N, n_iter), dtype=torch.float).to(device)
        else:
            c_time = None
        w_dist1 = tdist.Normal(
            torch.Tensor([0.0] * batch_size).to(device),
            torch.Tensor([1.0] * batch_size).to(device),
        )
        w_dist2 = tdist.Normal(
            torch.Tensor([0.0] * batch_size).to(device),
            torch.Tensor([1.0] * batch_size).to(device),
        )

        # Perform the solve over the specified number of iterations
        pump_rate = 1
        for i in range(n_iter):

            noise_ratio_i = 1.0
            if pump_rate_flag:
                pump_rate = (i + 1) / n_iter
                if (i + 1) / n_iter < 0.9:
                    noise_ratio_i = noise_ratio

            c_grads, s_grads = self.calculate_grads(c, s, q_mat, c_vector, p, pump_rate)
            W1t = w_dist1.sample((N,)).transpose(0, 1) * np.sqrt(lr) * noise_ratio_i
            W2t = w_dist2.sample((N,)).transpose(0, 1) * np.sqrt(lr) / noise_ratio_i
            c += lr * c_grads + 2 * g * torch.sqrt(c**2 + s**2 + 0.5) * W1t
            s += lr * s_grads + 2 * g * torch.sqrt(c**2 + s**2 + 0.5) * W2t

            if time_evolution_results:
                # Update the record of the values at each iteration with the values found
                # at this iteration
                c_time[:, :, i] = c

        # Ensure variables are within any problem constraints
        c = self.fit_to_constraints(c)

        # Stop the timer for the solve
        solve_time = time.time() - solve_time_start

        # Run the post processor on the results, if specified
        if post_processor:
            post_processor_object = PostProcessorFactory.create_postprocessor(
                post_processor
            )

            problem_variables = post_processor_object.postprocess(c, q_mat, c_vector)
            pp_time = post_processor_object.pp_time
        else:
            problem_variables = c
            pp_time = 0.0

        # Calculate the objective value
        # Perform a change of variables to enforce the box constraints
        confs = self.change_variables(problem_variables)
        objval = instance.compute_energy(confs)

        solution = Solution(
            problem_size=N,
            batch_size=batch_size,
            instance_name=instance.name,
            iter=n_iter,
            objective_value=objval,
            solve_time=solve_time,
            pp_time=pp_time,
            optimal_value=instance.optimal_sol,
            variables={
                "problem_variables": problem_variables,
                "s": s,
            },
            device=device,
        )

        return solution