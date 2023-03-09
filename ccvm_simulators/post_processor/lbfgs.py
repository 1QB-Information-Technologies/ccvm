from .post_processor import PostProcessor, MethodType
from .box_qp_model import BoxQPModel
import time
import torch
import tqdm


class PostProcessorLBFGS(PostProcessor):
    """A concrete class that implements the PostProcessor interface."""

    def __init__(self):
        self.pp_time = 0
        self.method_type = MethodType.LBFGS

    def postprocess(self, c, q_matrix, v_vector, num_iter=1):
        """Post processing using LBFGS method.

        Args:
            c (torch.tensor): The values for each
            variable of the problem in the solution found by the solver.
            q_matrix (torch.tensor): Coefficients of the quadratic terms.
            v_vector (torch.tensor): Coefficients of the linear terms.
            num_iter (int, optional): The number of iterations. Defaults to 1.

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
            (batch_size, size) = c.size()
        except Exception as e:
            raise e

        variables_pp = torch.zeros((batch_size, size))
        for bb in tqdm.tqdm(range(batch_size)):
            model = BoxQPModel(c[bb], self.method_type)
            for _ in range(num_iter):

                optimizer = torch.optim.LBFGS(model.parameters(), lr=0.001, max_iter=1)

                def closure():
                    optimizer.zero_grad()
                    loss = model(q_matrix, v_vector)
                    loss.backward()
                    return loss

                optimizer.step(closure)
                model.params = torch.nn.Parameter(torch.clamp(model.params, 0, 1))
            variables_pp[bb] = model.params.detach()
        end_time = time.time()
        self.pp_time = end_time - start_time
        return variables_pp
