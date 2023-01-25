from ccvm.solvers.ccvm_solver import CCVMSolver
from ccvm.post_processor.PostProcessorFactory import PostProcessorFactory
import torch
import numpy as np
import torch.distributions as tdist
import time

# The value used by the DLSolver when calculating a scaling value in super.get_scaling_factor()
DL_SCALING_MULTIPLIER = 0.5


class DLSolver(CCVMSolver):
    """The DLSolver class models the delay line coherent continuous-variable machine (DL-CCVM)."""

    def __init__(self, device, time_evolution_results=True, batch_size=1000):
        super().__init__(device)
        self.time_evolution_results = time_evolution_results
        self.batch_size = batch_size
        self._scaling_multiplier = DL_SCALING_MULTIPLIER

    def _validate_parameters(self, parameters):
        """Validate the parameter key against the keys in the expected parameters for DL solver.

                :param parameters: The set of parameters that will be used by the solver when solving problems. The parameters must match the format
               {
                   <problem size>: <dict with set of keys: p (pump), scale, lr (learning rate), iter (iterations), nr (noise_ratio)
               }
        For example:
                {
                    20: {"p": 2.0, "scale": None, "lr": 0.005, "iter": 15000, "nr": 10},
                    30: {"p": 2.0, "scale": None, "lr": 0.005, "iter": 15000, "nr": 5},
                }
                :type parameters: dict
                :raises ValueError: If the parameter key is not valid for this solver
        """
        expected_dlparameter_key_set = set(["p", "scale", "lr", "iter", "nr"])
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

    def compute_energy(self, confs, q_mat, c_vector, scaling_val):
        """Compute energy of configuration by xJx + hx formula

        :param confs: Configurations for which to compute energy
        :type confs: torch.Tensor
        :param q_mat: coefficients of the quadratic terms
        :type torch.Tensor
        :param c_vector:coefficients of the linear terms
        :type torch.Tensor
        :param scaling_val: scaling value of the coefficient
        :type scaling_val: float
        :return: Energy of configurations
        :rtype: torch.Tensor
        """
        confs_pow = 0.5 * (confs + 1)
        energy1 = (
            torch.einsum("bi, ij, bj -> b", confs_pow, q_mat, confs_pow) * scaling_val
        )
        energy2 = torch.einsum("bi, i -> b", confs_pow, c_vector) * scaling_val
        return 0.5 * energy1 + energy2

    def calculate_grads(self, c, s, q_matrix, c_vector, p, rate, S=1):
        """We treat the SDE that simulates the CIM of NTT as gradient
        calculation. Original SDE considers only quadratic part of the objective
        function. Therefore, we need to modify and add linear part of the QP to
        the SDE.

        :param c: amplitudes
        :type c: torch.Tensor
        :param s: _description_
        :type s: torch.Tensor
        :param q_matrix: coefficients of the quadratic terms
        :type q_matrix: torch.Tensor
        :param c_vector: coefficients of the linear terms
        :type c_vector: torch.Tensor
        :param p: _description_
        :type p: float
        :param rate: _description_
        :type rate: float
        :type S: int
        :param S: _description_
        :return: grads
        :rtype: torch.Tensor
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

    def tune(self, instances, post_processor=None, pump_rate_flag=True, g=0.05):
        """_summary_

        :param instances: A list of the instances that will be used to tune the solver parameters.
        :type instances: List[ccvm.problem.ProblemInstance]
        :param post_processor: _description_
        :type post_processor: PostProcessorType
        :param noise_ratio: _description_
        :type noise_ratio: _type_
        """
        # TODO: summary/descriptions
        # TODO: This implementation is a placeholder; full implementation is a
        #       future consideration
        self.is_tuned = True

    def solve(self, instance, post_processor=None, pump_rate_flag=True, g=0.05):
        """Solves the given problem instance using the DL-CCVM solver.

        :param instance: The problem to solve.
        :type instance: ccvm.problem.ProblemInstance
        :param post_processor: The post processor to use to process the results of the solver. None if no post processing is desired.
        :type post_processor: PostProcessorType
        :param pump_rate_flag: Whether or not to scale the pump rate based on the iteration number. If False, the pump rate will be 1.0.
        :type pump_rate_flag: bool
        :param g: _description_
        :type g: float
        :return: A dictionary containing the results of the solver. It contains these keys:
            - "c_variables" (:py:class:`torch.Tensor`) - The final values for each
            variable of the problem in the solution found by the solver
            - "c_evolution" (:py:class:`torch.Tensor`) - The values for each
            variable of the problem in the solution found by the solver in each
            iteration without post-processing
            - "objective_value" (:py:class:`torch.Tensor`) - The value of the objective function for the solution found by the solver
            - "solve_time" (float) - The time taken (in seconds) to solve the problem
            - "post_processing_time" (float) - The time taken (in seconds) to postprocess the solution
        :rtype: dict
        """
        # Get problem from problem instance
        n = instance.N
        q_mat = instance.q
        c_vector = instance.c

        # If the instance and the solver don't specify the same device type, move the tensors to the device type of the solver
        if instance.device != self.device:
            q_mat = q_mat.to(self.device)
            c_vector = c_vector.to(self.device)

        # Get solver setup variables
        batch_size = self.batch_size
        device = self.device
        time_evolution_results = self.time_evolution_results

        # Get parameters from parameter_key
        try:
            p = self.parameter_key[n]["p"]
            lr = self.parameter_key[n]["lr"]
            n_iter = self.parameter_key[n]["iter"]
            noise_ratio = self.parameter_key[n]["nr"]
        except KeyError as e:
            raise KeyError(
                f"The parameter '{e.args[0]}' for the given instance size is not defined."
            ) from e

        # Initialize tensor variables on the device that will be used to perform the calculations
        c = torch.zeros((batch_size, n), dtype=torch.float).to(device)
        s = torch.zeros((batch_size, n), dtype=torch.float).to(device)
        if time_evolution_results:
            c_time = torch.zeros((batch_size, n, n_iter), dtype=torch.float).to(device)
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
        solve_time_start = time.time()
        for i in range(n_iter):

            noise_ratio_i = 1.0
            if pump_rate_flag:
                pump_rate = (i + 1) / n_iter
                if (i + 1) / n_iter < 0.9:
                    noise_ratio_i = noise_ratio

            c_grads, s_grads = self.calculate_grads(c, s, q_mat, c_vector, p, pump_rate)
            W1t = w_dist1.sample((n,)).transpose(0, 1) * np.sqrt(lr) * noise_ratio_i
            W2t = w_dist2.sample((n,)).transpose(0, 1) * np.sqrt(lr) / noise_ratio_i
            c += lr * c_grads + 2 * g * torch.sqrt(c**2 + s**2 + 0.5) * W1t
            s += lr * s_grads + 2 * g * torch.sqrt(c**2 + s**2 + 0.5) * W2t

            if time_evolution_results:
                # Update the record of the values at each iteration with the values found at this iteration
                c_time[:, :, i] = c
        solve_time = time.time() - solve_time_start

        # Clip the amplitudes
        c = torch.clamp(c, -1, 1)
        s = torch.clamp(s, -1, 1)  # TODO: this is not used

        # Run the post processor on the results, if specified
        if post_processor:
            post_processor_object = PostProcessorFactory.create_postprocessor(
                post_processor
            )

            c_variables = post_processor_object.postprocess(c, q_mat, c_vector)
            pp_time = post_processor_object.pp_time
        else:
            c_variables = c
            pp_time = 0.0

        # Calculate the objective value
        objval = self.compute_energy(c_variables, q_mat, c_vector, instance.scaled_by)

        return {
            "c_variables": c_variables,
            "c_evolution": c_time,
            "objective_value": objval,
            "solve_time": solve_time,
            "post_processing_time": pp_time,
        }
