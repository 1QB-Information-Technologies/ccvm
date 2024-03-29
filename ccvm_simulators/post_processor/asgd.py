from .post_processor import PostProcessor, MethodType
from .box_qp_model import BoxQPModel
import time
import torch
import tqdm


class PostProcessorASGD(PostProcessor):
    """A concrete class that implements the PostProcessor interface."""

    def __init__(self):
        self.pp_time = 0
        self.method_type = MethodType.ASGD

    def postprocess(
        self,
        c,
        q_matrix,
        v_vector,
        lower_clamp=0.0,
        upper_clamp=1.0,
        num_iter=1,
        device="cpu",
    ):
        """Post processing using ASGD method.

        Args:
            c (torch.tensor): The values for each
            variable of the problem in the solution found by the solver.
            q_matrix (torch.tensor): The Q matrix describing the BoxQP problem.
            v_vector (torch.tensor): The V vector describing the BoxQP problem.
            lower_clamp (float, optional): Lower bound of the box constraints. Defaults
                to 0.0.
            upper_clamp (float, optional): Upper bound of the box constraints. Defaults
                to 1.0.
            num_iter (int, optional): The number of iterations. Defaults to 1.
            device (str, optional): Defines which GPU (or the CPU) to use.
                Defaults to "cpu".

        Returns:
            torch.tensor: The values for each variable of the problem in the
                solution found by the solver after post-processing.
        """
        start_time = time.time()

        try:
            if not torch.is_tensor(c):
                raise TypeError("parameter c must be a tensor")
            if not torch.is_tensor(q_matrix):
                raise TypeError("parameter q_matrix must be a tensor")
            if not torch.is_tensor(v_vector):
                raise TypeError("parameter v_vector must be a tensor")
            (batch_size, _) = c.size()
            model = BoxQPModel(c, self.method_type)
        except Exception as e:
            raise e

        optimizer = torch.optim.ASGD(model.parameters(), lr=0.01, lambd=0.001)
        for _ in tqdm.tqdm(range(num_iter)):
            loss = model(q_matrix, v_vector)
            loss.backward(torch.Tensor([1] * batch_size).to(device))
            optimizer.step()
            optimizer.zero_grad()
            model.params = torch.nn.Parameter(
                torch.clamp(model.params, lower_clamp, upper_clamp)
            )
        end_time = time.time()
        self.pp_time = end_time - start_time
        return model.params.detach()
