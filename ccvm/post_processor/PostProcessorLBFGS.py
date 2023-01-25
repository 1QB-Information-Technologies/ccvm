from ccvm.post_processor.PostProcessor import PostProcessor
from ccvm.post_processor.utils import func_post, func_post_LBFGS, func_post_jac
import numpy as np
import time
import torch
import tqdm


class PostProcessorLBFGS(PostProcessor):
    """A concrete class that implements the PostProcessor interface."""

    def __init__(self):
        self.pp_time = 0

    def postprocess(self, c, q_mat, c_vector, optim_iter=1, device="cpu"):
        """Post processing using LBFGS method.

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
        (batch_size, size) = c.size()
        variables_pp = torch.zeros((batch_size, size))
        for bb in tqdm.tqdm(range(batch_size)):
            model = BoxQP_LBFGS(c[bb])
            for _ in range(optim_iter):

                optimizer = torch.optim.LBFGS(model.parameters(), lr=0.001, max_iter=1)

                def closure():
                    optimizer.zero_grad()
                    loss = model(q_mat, c_vector)
                    loss.backward()
                    return loss

                optimizer.step(closure)
                model.params = torch.nn.Parameter(torch.clamp(model.params, 0, 1))
            variables_pp[bb] = model.params.detach()
        end_time = time.time()
        self.pp_time = end_time - start_time
        return variables_pp


class BoxQP_LBFGS(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.params = torch.nn.Parameter(c)

    def forward(self, q_mat, c_vector):
        c_variables = self.params
        return func_post_LBFGS(c_variables, q_mat, c_vector)
