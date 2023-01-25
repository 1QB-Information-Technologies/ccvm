from ccvm.post_processor.PostProcessor import PostProcessor
from ccvm.post_processor.utils import func_post, func_post_hess, func_post_jac
import numpy as np
import time
import torch
import tqdm


class PostProcessorTrustConstr(PostProcessor):
    """A concrete class that implements the PostProcessor interface."""

    def __init__(self):
        self.pp_time = 0

    def postprocess(self, c, q_mat, c_vector, optim_iter=1, device="cpu"):
        """Post processing using TrustConstr method.

        :param c:
        :type Tensor
        :param q_mat: coefficients of the quadratic terms
        :type Tensor
        :param c_vector: coefficients of the linear terms
        :type Tensor
        :param optim_iter:
        :type int
        :param device:
        :type str, defaults to cpu
        :return: c_variables
        :rtype: Tensor
        """
        start_time = time.time()
        c = np.array(c.cpu())
        batch_size = np.shape(c)[0]
        size = np.shape(c)[1]
        bounds = ((0, 1.0),) * size
        c_variables = np.zeros((batch_size, size))
        for bb in tqdm.tqdm(range(batch_size)):
            c0 = c[bb]
            res = minimize(
                func_post,
                c0,
                args=(q_mat, c_vector),
                method="trust-constr",
                bounds=bounds,
                jac=func_post_jac,
                hess=func_post_hess,
                options={"maxiter": 50, "gtol": 1e-6},
            )
            c_variables[bb] = res.x
        end_time = time.time()
        self.pp_time = end_time - start_time
        return torch.Tensor(c_variables).to(device)
