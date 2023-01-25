from ccvm.post_processor.PostProcessor import PostProcessor
from ccvm.post_processor.utils import BoxQP
import time
import torch
import tqdm


class PostProcessorASGD(PostProcessor):
    """A concrete class that implements the PostProcessor interface."""

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
        :return: c_variables
        :rtype: Tensor
        """
        start_time = time.time()
        (batch_size, size) = c.size()
        Model = BoxQP(c)
        optimizer = torch.optim.ASGD(Model.parameters(), lr=0.01, lambd=0.001)
        for _ in tqdm.tqdm(range(optim_iter)):
            loss = Model(q_mat, c_vector)
            loss.backward(torch.Tensor([1] * batch_size).to(device))
            optimizer.step()
            optimizer.zero_grad()
            Model.params = torch.nn.Parameter(torch.clamp(Model.params, 0, 1))
        end_time = time.time()
        self.pp_time = end_time - start_time
        return Model.params.detach()
