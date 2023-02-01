from ccvm.post_processor.PostProcessor import PostProcessor, MethodType
from ccvm.post_processor.BoxQPModel import BoxQPModel
import time
import torch
import tqdm


class PostProcessorAdam(PostProcessor):
    """A concrete class that implements the PostProcessor interface."""

    def __init__(self):
        self.pp_time = 0
        self.method_type = MethodType.Adam

    def postprocess(self, c, q_mat, c_vector, num_iter=1, device="cpu"):
        """Post processing using Adam method.

        Args:
            c (torch.tensor): The values for each
            variable of the problem in the solution found by the solver.
            q_mat (torch.tensor): Coefficients of the quadratic terms.
            c_vector (torch.tensor): Coefficients of the linear terms.
            num_iter (int, optional): The number of iterations. Defaults to 1.
            device (str, optional): Defines which GPU (or the CPU) to use.
                Defaults to "cpu".

        Returns:
            torch.tensor: The values for each variable of the problem in
                the solution found by the solver after post-processing.
        """
        start_time = time.time()

        try:
            if not torch.is_tensor(c):
                raise TypeError("parameter c must be a tensor")
            if not torch.is_tensor(q_mat):
                raise TypeError("parameter q_mat must be a tensor")
            if not torch.is_tensor(c_vector):
                raise TypeError("parameter c_vector must be a tensor")
            (batch_size, _) = c.size()
            model = BoxQPModel(c, self.method_type)
        except Exception as e:
            raise e

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
        for _ in tqdm.tqdm(range(num_iter)):
            loss = model(q_mat, c_vector)
            loss.backward(torch.Tensor([1] * batch_size).to(device))
            optimizer.step()
            optimizer.zero_grad()
            model.params = torch.nn.Parameter(torch.clamp(model.params, 0, 1))
        end_time = time.time()
        self.pp_time = end_time - start_time
        return model.params.detach()
