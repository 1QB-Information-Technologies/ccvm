from abc import ABC, abstractmethod, abstractproperty
import torch
import enum


class DeviceType(enum.Enum):
    CPU_DEVICE = "cpu"
    CUDA_DEVICE = "cuda"


class PostProcessorType:
    # TODO: Move this to the file with the post processor class
    BFGS = "BFGS"
    ADAM = "Adam"
    LBFGS = "LBFGS"
    ASGD = "ASGD"
    TRUST_CONSTR = "trust-constr"


class CCVMSolver(ABC):
    """Constructor method

    :param device: Defines which GPU (or the CPU) to use.
    :type device: DeviceType
    """

    def __init__(self, device):
        if device not in DeviceType._value2member_map_:
            raise ValueError("Given device is not available")
        self.device = device
        self._scaling_multiplier = 1
        self._is_tuned = False
        self._parameter_key = None

    ##################################
    # Properties                     #
    ##################################
    @property
    def is_tuned(self):
        """Returns true if the current solver parameters were set by the tune() function.
        :returns: is_tuned
        :rtype: bool
        """
        return self._is_tuned

    @property
    def parameter_key(self):
        """The set of parameters that will be used by the solver when solving the problem.
        :returns: parameter_key
        :rtype: dict
        """
        return self._parameter_key

    @abstractmethod
    def _validate_parameters(self, parameters):
        """Validates the parameters to make sure the values in the dictionary are
           compatible with the solver

        :param parameters: The parameters intended to be used by the solver when solving
        :type parameters: dict
        """
        pass

    @parameter_key.setter
    def parameter_key(self, parameters):
        """Manually set the value of the parameters that the solver will use when solving problems.
           Note: setting this parameter after calling tune() will overwrite tuned parameters.

        :param parameters: The set of parameters that will be used by the solver when solving the problem.
        :type parameters: dict
        """
        self._validate_parameters(parameters)
        self._parameter_key = parameters
        self._is_tuned = False

    ##################################
    # Abstract Methods               #
    ##################################
    @abstractmethod
    def calculate_grads(self):
        # TODO: description
        pass

    @abstractmethod
    def compute_energy(self):
        # TODO: description
        pass

    @abstractmethod
    def calculate_grads(self):
        # TODO: description
        pass

    @abstractmethod
    def tune(self):
        # TODO: description
        pass

    @abstractmethod
    def solve(self):
        # TODO: description
        pass

    ##################################
    # Implemented Methods            #
    ##################################

    def get_scaling_factor(self, N, q):
        """Reads the `scale` value from the parameter_key for a problem of size N,
        or uses a default calculation if provided scaling value is None.
        :param N: the size of the instance to be scaled
        :type N: int
        :param q: the quadratic coefficients of the instance to be scaled
        :type q: torch.Tensor
        """
        try:
            if "scale" in self.parameter_key[N] and self.parameter_key[N]["scale"] is not None:
                # Read the scaling value from the parameter key
                scaling_val = self.parameter_key[N]["scale"]
            else:
                # Calculate the scaling value from the problem's quadratic terms
                scaling_val = torch.sqrt(torch.sum(torch.abs(q))) * self._scaling_multiplier
        except KeyError as e:
            raise Exception(
                f"The solver's parameter_key does not contain values for problem size N = {N}"
            ) from e
        return scaling_val
