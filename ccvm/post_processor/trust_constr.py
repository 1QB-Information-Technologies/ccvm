from .post_processor import PostProcessor
from scipy.optimize import minimize
import numpy as np
import time
import torch
import tqdm


class PostProcessorTrustConstr(PostProcessor):
    """A concrete class that implements the PostProcessor interface."""

    def __init__(self):
        self.pp_time = 0

    def postprocess(self, c, q_matrix, v_vector):
        """Post processing using TrustConstr method.

        Args:
            c (torch.tensor): The values for each
            variable of the problem in the solution found by the solver.
            q_matrix (torch.tensor): The Q matrix describing the BoxQP problem.
            v_vector (torch.tensor): The V vector describing the BoxQP problem.

        Returns:
            torch.tensor: The values for each variable of the problem
            in the solution found by the solver after post-processing.
        """
        start_time = time.time()
        try:
            if not torch.is_tensor(c):
                raise TypeError("parameter c must be a tensor")
            if not torch.is_tensor(q_matrix):
                raise TypeError("parameter q_matrix must be a tensor")
            if not torch.is_tensor(v_vector):
                raise TypeError("parameter v_vector must be a tensor")
            c = np.array(c.cpu())
            batch_size = np.shape(c)[0]
            size = np.shape(c)[1]
            bounds = ((0, 1.0),) * size
            c_variables = np.zeros((batch_size, size))
        except Exception as e:
            raise e
        for bb in tqdm.tqdm(range(batch_size)):
            c0 = c[bb]
            res = minimize(
                super().func_post,
                c0,
                args=(q_matrix, v_vector),
                method="trust-constr",
                bounds=bounds,
                jac=super().func_post_jac,
                hess=self.func_post_hess,
                options={"maxiter": 50, "gtol": 1e-6},
            )
            c_variables[bb] = res.x
        end_time = time.time()
        self.pp_time = end_time - start_time
        return torch.Tensor(c_variables)

    def func_post_hess(c, *args):
        """Calculates the Hessian of the objective function as a numpy
            matrix. Providing it can improve the performance but it can
            only be used with the numpy "trust-constr" post-processing method.

        Args:
            c (torch.tensor): The values for each variable of the problem
                in the solution found by the solver.

        Returns:
            torch.tensor: objective function.
        """
        q_matrix = np.array(args[0].cpu())
        return 0.5 * q_matrix
