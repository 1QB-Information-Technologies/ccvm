from ccvm.solvers.ccvm_solver import CCVMSolver
from ccvm.solution import Solution
from ccvm.post_processor.factory import PostProcessorFactory
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
        time_evolution_results=False,
        batch_size=1000,
    ):
        """
        Args:
            device (str): The device to use for the solver. Can be "cpu" or "cuda".
            problem_category (str): The category of problem to solve. Can be one
            of "boxqp". Defaults to "boxqp".
            time_evolution_results (bool): Whether to return the time evolution
            results for each iteration during the solve. Defaults to True.
            batch_size (int): The batch size of the problem. Defaults to 1000.

        Raises:
            ValueError: If the problem category is not supported by the solver.

        Returns:
            DLSolver: The DLSolver object.
        """
        super().__init__(device)
        self.time_evolution_results = time_evolution_results
        self.batch_size = batch_size
        self._scaling_multiplier = MF_SCALING_MULTIPLIER
        # Use the method selector to choose the problem-specific methods to use
        self._method_selector(problem_category)

    def _validate_parameters(self, parameters):
        """Validate the parameter key against the keys in the expected parameters
        for MF solver.

        Args:
            parameters (dict): The parameters to validate. The parameters must
            match the format:
                {
                    <problem size>:
                        <dict with set of keys:
                            pump,
                            feedback_scale,
                            j (measurement strength),
                            S (enforced saturation value), can be scalar or a vector of size 'problem_size'.
                            lr (learning rate),
                            iterations>
                }
            For example:
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
            ValueError: If the parameter key is not valid for this solver.
        """
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
        mu_tilde_pow = torch.pow(mu_tilde, 2) / S**2

        mu_term1 = (-(1 + j) + pump - g**2 * mu_pow) * mu
        mu_term2_1 = (
            -(torch.einsum("bi,ij -> bj", mu_tilde_pow, q_matrix)) * mu_tilde / S
        )
        mu_term2_2 = -torch.einsum("j,bj -> bj", v_vector, mu_tilde / S)
        mu_term3 = np.sqrt(j) * (sigma - 0.5) * wiener_increment

        sigma_term1 = 2 * (-(1 + j) + pump - 3 * g**2 * mu_pow) * sigma
        sigma_term2 = -2 * j * (sigma - 0.5).pow(2)
        sigma_term3 = (1 + j) + 2 * g**2 * mu_pow

        grads_mu = mu_term1 + fs * (mu_term2_1 + mu_term2_2) + mu_term3
        grads_sigma = sigma_term1 + sigma_term2 + sigma_term3

        return grads_mu, grads_sigma

    def _change_variables_boxqp(self):
        # MFSolver does not require a change of variables when solving the boxqp
        # problem
        pass

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

    def tune(self, instances, post_processor, g=0.01):
        """Determines the best parameters for the solver to use by adjusting each
        parameter over a number of iterations on the problems in the given set of
        problems instances. The `parameter_key` attribute of the solver will be
        updated with the best parameters found.

        Args:
            instances (list): A list of problem instances to tune the solver on.
            post_processor (str): The name of the post processor to use to process
            the results of the solver. None if no post processing is desired.
            Defaults to None.
            g (float): The nonlinearity coefficient. Defaults to 0.01.
        """
        # TODO: This implementation is a placeholder; full implementation is a
        #       future consideration
        self.is_tuned = True

    def solve(self, instance, post_processor=None, g=0.01, pump_rate_flag=True):
        """Solves the given problem instance using the tuned or specified
        parameters in the parameter key.

        Args:
            instance (boxqp.boxqp.ProblemInstance): The problem to solve.
            post_processor (str): The name of the post processor to use to process
            the results of the solver. None if no post processing is desired.
            g (float, optional): The nonlinearity coefficient. Defaults to 0.01
            pump_rate_flag (bool, optional): Whether or not to scale the pump rate based
            on the iteration number. If False, the pump rate will be 1.0. Defaults to
            True.

        Returns:
            dict: A dictionary containing the results of the solver. It contains
            these keys:
            - "problem_variables" (:py:class:`torch.Tensor`) - The final values for each
            variable of the problem in the solution found by the solver
            - "mu_evolution" (:py:class:`torch.Tensor`) - The values of mu at each
            iteration during the solve process
            - "sigma_evolution" (:py:class:`torch.Tensor`) - The values of sigma at
            each iteration during the solve process
            - "objective_values" (:py:class:`torch.Tensor`) - The value of the objective
            function for the solution found by the solver
            - "solve_time" (float) - The time taken (in seconds) to solve the problem
            - "post_processing_time" (float) - The time taken (in seconds) to postprocess
            the solution
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
        time_evolution_results = self.time_evolution_results

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

        # Initialize tensor variables on the device that will be used to perform
        # the calculations
        mu = torch.zeros((batch_size, problem_size), dtype=torch.float).to(device)
        sigma = torch.ones((batch_size, problem_size), dtype=torch.float).to(device) * (
            1 / 4
        )
        mu_time = sigma_time = None
        if time_evolution_results:
            mu_time = torch.zeros(
                (batch_size, problem_size, iterations), dtype=torch.float
            ).to(device)
            sigma_time = torch.zeros(
                (batch_size, problem_size, iterations), dtype=torch.float
            ).to(device)

        wiener_dist = tdist.Normal(
            torch.Tensor([0.0] * batch_size).to(device),
            torch.Tensor([1.0] * batch_size).to(device),
        )

        # Perform the solve over the specified number of iterations
        pump_rate = 1
        for i in range(iterations):

            wiener = wiener_dist.sample((problem_size,)).transpose(0, 1)
            wiener_increment = wiener / np.sqrt(lr)
            mu_tilde = mu + np.sqrt(1 / (4 * j)) * wiener_increment
            mu_tilde_c = self.fit_to_constraints(mu_tilde, -S, S)

            if pump_rate_flag:
                pump_rate = (i + 1) / iterations

            if (i + 1) / iterations < 0.8:
                j_i = j
            else:
                j_i = 0.1

            instantaneous_pump = pump * pump_rate

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

            if time_evolution_results:
                # Update the record of the values at each iteration with the
                # values found at this iteration
                mu_time[:, :, i] = mu
                sigma_time[:, :, i] = sigma

        mu_tilde = self.fit_to_constraints(mu_tilde, -S, S)

        solve_time = time.time() - solve_time_start

        # Run the post processor on the results, if specified
        if post_processor:
            post_processor_object = PostProcessorFactory.create_postprocessor(
                post_processor
            )

            problem_variables = post_processor_object.postprocess(
                mu_tilde.pow(2) / S**2, q_matrix, v_vector, device=device
            )
            pp_time = post_processor_object.pp_time
        else:
            problem_variables = mu_tilde.pow(2) / S**2
            pp_time = 0.0

        objval = instance.compute_energy(problem_variables)

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

        return solution
