from .post_processor import PostProcessor
from scipy.optimize import minimize
import numpy as np
import time
import torch
import tqdm


class PostProcessorBFGS(PostProcessor):
    def __init__(self):
        self.pp_time = 0

    def postprocess(self, c, q_matrix, v_vector):
        """Post processing using BFGS method.

        Args:
            c (torch.tensor): The values for each
            variable of the problem in the solution found by the solver.
            q_matrix (torch.tensor): Coefficients of the quadratic terms.
            v_vector (torch.tensor): Coefficients of the linear terms.

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
            c0 = 0.5 * (c[bb] + 1)
            res = minimize(
                super().func_post,
                c0,
                args=(q_matrix, v_vector),
                method="L-BFGS-B",
                bounds=bounds,
                jac=super().func_post_jac,
            )
            c_variables[bb] = res.x
        c_variables = 2 * (c_variables - 0.5)
        end_time = time.time()
        self.pp_time = end_time - start_time
        return torch.Tensor(c_variables)
