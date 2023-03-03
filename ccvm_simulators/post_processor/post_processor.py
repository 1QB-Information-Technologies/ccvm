from abc import ABC, abstractmethod
from enum import Enum
import numpy as np


class MethodType(str, Enum):
    BFGS = "bfgs"
    TrustConst = "trust-constr"
    LBFGS = "lbfgs"
    Adam = "adam"
    ASGD = "asgd"


class PostProcessor(ABC):
    """A Hypothetical PostProcessor Class Interface."""

    @abstractmethod
    def postprocess(self):
        """An abstract interface method."""
        pass

    def func_post(self, c, *args):
        """Generates the objective function as a numpy scalar.

        Args:
            c (torch.tensor): The values for each
            variable of the problem in the solution found by the solver.

        Returns:
            torch.tensor: Objective function.
        """
        q_matrix = np.array(args[0].cpu())
        v_vector = np.array(args[1].cpu())
        energy1 = np.einsum("i, ij, j", c, q_matrix, c)
        energy2 = np.einsum("i, i", c, v_vector)
        return 0.5 * energy1 + energy2

    def func_post_jac(self, c, *args):
        """Calculates the Jacobian of the objective function as a numpy vector
        for the post-processing if the post processing is performed using the
        Jacobian. The post-processing can still be performed without the
        Jacobian but having it for some post-processing methods can improve the
        performance of the post-processing. Jacobian can only be used with the
        numpy post-processing methods of "BFGS" and "trust-constr".

        Args:
            c (torch.tensor): The values for each
            variable of the problem in the solution found by the solver.

        Returns:
            torch.tensor: Objective function.
        """
        q_matrix = np.array(args[0].cpu())
        v_vector = np.array(args[1].cpu())
        energy1_jac = np.einsum("ij,j->i", q_matrix, c)
        energy2_jac = v_vector
        return energy1_jac + energy2_jac
