from abc import ABC, abstractmethod, abstractproperty
import torch
import enum


class DeviceType(enum.Enum):
    """The devices that can be used by pytorch"""

    CPU_DEVICE = "cpu"
    CUDA_DEVICE = "cuda"


class CCVMSolver(ABC):
    """The base class for all solvers. This class should not be used directly; one of
    the subclasses should be used.

    Args:
        device (DeviceType): The device that the solver will use to solve the problem.
    """

    def __init__(self, device):
        if device not in DeviceType._value2member_map_:
            raise ValueError("Given device is not available")
        self.device = device
        self._is_tuned = False
        self._scaling_multiplier = None
        self._parameter_key = None
        self.calculate_grads = None
        self.change_variables = None
        self.fit_to_constraints = None

    ##################################
    # Properties                     #
    ##################################
    @property
    def is_tuned(self):
        """bool: True if the current solver parameters were set by the tune() function."""
        return self._is_tuned

    @property
    def parameter_key(self):
        """The parameters that will be used by the solver when solving the problem.

        Note:
            Setting this parameter after calling tune() will overwrite tuned parameters.

        This method should be overwritten by the subclass to ensure that the docstrings
        in the final documentation are personalized for the subclassed solver.

        The setter for this parameter must also be implemented in the subclass.

        Returns:
            dict: The parameter key.
        """
        return self._parameter_key

    ##################################
    # Abstract Methods               #
    ##################################

    @abstractmethod
    def tune(self):
        """Determines the best parameters for the solver to use by adjusting each
        parameter over a number of iterations on the problems in the given set of
        problems instances. The `parameter_key` attribute of the solver will be
        updated with the best parameters found.
        Input parameters to this function are specific to each solver.
        """
        pass

    @abstractmethod
    def solve(self):
        """Solves a given problem instance using the parameters in the solver's
        `parameter_key`
        """
        pass

    @abstractmethod
    def _calculate_grads_boxqp(self, **kwargs):
        """Calculates the gradients of the variables for the boxqp problem."""
        pass

    @abstractmethod
    def _change_variables_boxqp(self, **kwargs):
        """Performs a change of variables on the boxqp problem."""
        pass

    @abstractmethod
    def _fit_to_constraints_boxqp(self, **kwargs):
        """Fits the variables to the constraints for the boxqp problem."""
        pass

    ##################################
    # Implemented Methods            #
    ##################################

    def get_scaling_factor(self, q_matrix):
        """Uses a default calculation to determine the amount by which the problem
        coefficients should be scaled. The value may differ depending on the solver,
        as some solvers have different scaling multipliers.

        Args:
            q_matrix (torch.tensor): The Q matrix describing the BoxQP problem

        Returns:
            float: The recommended scaling factor to be use to scale the problem for this solver
        """
        # Calculate the scaling value from the problem's quadratic terms
        scaling_val = (
            torch.sqrt(torch.sum(torch.abs(q_matrix))) * self._scaling_multiplier
        )
        return scaling_val
