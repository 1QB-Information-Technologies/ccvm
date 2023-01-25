from ccvm.post_processor.PostProcessor import PostProcessor
from ccvm.post_processor.utils import func_post, func_post_jac
import numpy as np
import time
import torch


class PostProcessorBFGS(PostProcessor):
    def __init__(self):
        self.pp_time = 0

    def postprocess(self, c, q_mat, c_vector, optim_iter=1, device="cpu"):
        """Post processing using ASGD method.

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
            c0 = 0.5 * (c[bb] + 1)
            res = minimize(
                func_post,
                c0,
                args=(q_mat, c_vector),
                method="L-BFGS-B",
                bounds=bounds,
                jac=func_post_jac,
            )
            c_variables[bb] = res.x
        c_variables = 2 * (c_variables - 0.5)
        end_time = time.time()
        self.pp_time = end_time - start_time
        return torch.Tensor(c_variables).to(device)
