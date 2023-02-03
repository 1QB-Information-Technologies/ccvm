from .post_processor import MethodType
import torch


class BoxQPModel(torch.nn.Module):
    """Generates the model required for post-processing using torch. Utilizing
    the Adam or ASGD or LBFGS optimization methods by calling the function
    func_post_torch or func_post_LBFGS.

    Args:
        torch.nn.Module: Base class for all neural network modules.
    """

    def __init__(self, c, method_type):
        """BoxQPModel class initialization.

        Args:
            c (torch.tensor): The values for each
            variable of the problem in the solution found by the solver.
            method_type (MethodType): The type of method to be used in post-processing.
        """
        super().__init__()
        self.params = torch.nn.Parameter(c)
        self.method_type = method_type

    def forward(self, q_matrix, v_vector):
        """The forward method is called when we use the neural network to make a
        prediction. The forward method is called from the __call__ function of
        nn.Module, so that when we run model(input), the forward method is
        called.

        Args:
            q_matrix (torch.tensor): The Q matrix describing the BoxQP problem
            v_vector (torch.tensor): The V vector describing the BoxQP problem.

        Returns:
            torch.tensor: Objective function.
        """
        try:
            if not torch.is_tensor(q_matrix):
                raise TypeError("parameter q_matrix must be a tensor")
            if not torch.is_tensor(v_vector):
                raise TypeError("parameter v_vector must be a tensor")
        except Exception as e:
            raise e
        c_variables = self.params
        method_type = self.method_type
        if method_type == MethodType.LBFGS:
            return self.func_post_LBFGS(c_variables, q_matrix, v_vector)
        elif method_type == MethodType.Adam or method_type == MethodType.ASGD:
            return self.func_post_torch(c_variables, q_matrix, v_vector)
        else:
            raise ValueError(
                f"""Invalid method type provided for generating the model.
                Provided: {method_type}. Valid methods are {MethodType.Adam},
                {MethodType.ASGD} and {MethodType.LBFGS}."""
            )

    def func_post_torch(self, c, q_matrix, v_vector):
        """Generates the objective function as vector torch object. This
        should be used when post-processing in parallel for all batches.

        Args:
            c (torch.tensor): The values for each
            variable of the problem in the solution found by the solver.
            q_matrix (torch.tensor): The Q matrix describing the BoxQP problem
            v_vector (torch.tensor): The V vector describing the BoxQP problem.

        Returns:
            torch.tensor: Objective function.
        """
        energy1 = torch.einsum("bi, ij, bj -> b", c, q_matrix, c)
        energy2 = torch.einsum("bi, i -> b", c, v_vector)
        return 0.5 * energy1 + energy2

    def func_post_LBFGS(self, c, q_matrix, v_vector):
        """Generates the objective function as a scalar torch object. This
        should be used when post-processing for each batch separately.

        Args:
            c (torch.tensor): The values for each
            variable of the problem in the solution found by the solver.
            q_matrix (torch.tensor): The Q matrix describing the BoxQP problem
            v_vector (torch.tensor): The V vector describing the BoxQP problem.

        Returns:
            torch.tensor: Objective function.
        """
        energy1 = torch.einsum("i, ij, j", c, q_matrix, c)
        energy2 = torch.einsum("i, i", c, v_vector)
        return 0.5 * energy1 + energy2
