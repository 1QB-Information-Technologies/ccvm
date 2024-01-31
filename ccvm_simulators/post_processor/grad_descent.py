from .post_processor import PostProcessor
import time
import torch
import tqdm


class PostProcessorGradDescent(PostProcessor):
    # TODO: Issue 156 will allow constructor to take in num_iter_main.
    def __init__(self):
        """Initialize the PostProcessorGradDescent class."""
        self.pp_time = 0

    def postprocess(
        self,
        c,
        q_matrix,
        v_vector,
        lower_clamp=0.0,
        upper_clamp=1.0,
        num_iter_main=100,
        num_iter_pp=None,
        step_size=0.1,
    ):
        """Post processing using Gradient Descent method.

        Args:
            c (torch.tensor): The vector of initial values of the variables for the
                post-processor.
            q_matrix (torch.tensor): The Q matrix describing the BoxQP problem.
            v_vector (torch.tensor): The V vector describing the BoxQP problem.
            lower_clamp (float, optional): Lower bound of the box constraints. Defaults
                to 0.0.
            upper_clamp (float, optional): Upper bound of the box constraints. Defaults
                to 1.0.
            num_iter_main (int): The number of iterations for the main stochastic
                process. Defaults to 100.
            num_iter_pp (int, optional): The number of iterations for post-processing.
            Defaults to None, in which case it is set to one percent of the number of
            iterations for the main stochastic process.
            step_size (float, optional): Step size for the gradient descent. Defaults to
                0.1.

        Returns:
            torch.tensor: The vector of variables found using the post-processing method.
        """
        start_time = time.time()

        try:
            if not torch.is_tensor(c):
                raise TypeError("parameter c must be a tensor")
            if not torch.is_tensor(q_matrix):
                raise TypeError("parameter q_matrix must be a tensor")
            if not torch.is_tensor(v_vector):
                raise TypeError("parameter v_vector must be a tensor")
        except Exception as e:
            raise e

        if num_iter_pp is None:
            num_iter_pp = int(num_iter_main * 0.01)

        for _ in tqdm.tqdm(range(num_iter_pp)):
            c_grads = torch.einsum("bi,ij -> bj", c, q_matrix) + v_vector
            c += -step_size * c_grads
            c = torch.clamp(c, lower_clamp, upper_clamp)

        end_time = time.time()
        self.pp_time = end_time - start_time
        return c
