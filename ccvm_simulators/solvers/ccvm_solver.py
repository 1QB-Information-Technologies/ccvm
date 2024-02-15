from abc import ABC, abstractmethod
import torch
import enum
import numpy as np
from pandas import DataFrame


class DeviceType(enum.Enum):
    """The devices that can be used by pytorch"""

    CPU_DEVICE = "cpu"
    CUDA_DEVICE = "cuda"


class MachineType(enum.Enum):
    """The type of machine we are simulating."""

    CPU = "cpu"
    CUDA = "cuda"
    FPGA = "fpga"
    DL_CCVM = "dl-ccvm"
    MF_CCVM = "mf-ccvm"


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
        self._default_cpu_machine_parameters = {
            "cpu_power": {20: 4.93, 30: 5.19, 40: 5.0, 50: 5.01, 60: 5.0, 70: 5.22}
        }
        self._default_cuda_machine_parameters = {
            "gpu_power": {
                20: 28.93,
                30: 29.8,
                40: 31.09,
                50: 31.29,
                60: 31.49,
                70: 32.28,
            }
        }
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
    def _solve(self):
        """Solves a given problem instance using the parameters in the solver's
        `parameter_key`
        """
        pass

    @abstractmethod
    def _solve_adam(self):
        """Solves a given problem instance with an enhancement of Adam algorithm
        using the parameters in the solver's `parameter_key`
        """
        pass

    @abstractmethod
    def _calculate_drift_boxqp(self, **kwargs):
        """Calculates the drift part of the CCVM for the boxqp problem."""
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

    def _method_selector(self, problem_category):
        """Set methods relevant to this category of problem

        Args:
            problem_category (str): The category of problem to solve. Can be one of "boxqp".

        Raises:
            ValueError: If the problem category is not supported by the solver.
        """
        if problem_category.lower() == "boxqp":
            self.calculate_drift = self._calculate_drift_boxqp
            self.calculate_grads = self._calculate_grads_boxqp
            self.change_variables = self._change_variables_boxqp
            self.fit_to_constraints = self._fit_to_constraints_boxqp
        else:
            raise ValueError(
                "The given instance is not a valid problem category."
                f" Given category: {problem_category}"
            )

    def _validate_dataframe_columns(self, dataframe):
        """Validates that the given dataframe contains the required columns when
        calculating optics machine energy on DL-CCVM and MF-CCVM solvers.

        Args:
            dataframe (DataFrame): The dataframe to validate.

        Raises:
            ValueError: If the given dataframe is missing any of the required columns.
        """
        required_columns = ["pp_time", "iterations"]

        missing_columns = [
            col for col in required_columns if col not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"The given dataframe is missing the following columns: {missing_columns}"
            )

    def cpu_energy_max(self, machine_parameters: dict = None):
        """The wrapper function of calculating the maximum energy consumption of the
        solver simulating on a CPU machine.

        Args:
            machine_parameters (dict, optional): Parameters of the. Defaults to None.

        Raises:
            ValueError: when the given machine parameters are not valid.
            ValueError: when the given dataframe does not contain the required columns.

        Returns:
            Callable: A callable function that takes in a dataframe and problem size and
                returns the maximum energy consumption of the solver.
        """
        if machine_parameters is None:
            machine_parameters = self._default_cpu_machine_parameters
        else:
            if "cpu_power" not in machine_parameters.keys():
                raise ValueError(
                    "The given machine parameters are not valid. "
                    "The dictionary must contain the key 'cpu_power'"
                )

        def cpu_energy_max_callable(matching_df: DataFrame, problem_size: int):
            """Calculate the maximum power consumption of the solver simulating on a cpu
                machine.

            Args:
                matching_df (DataFrame): The necessary data to calculate the maximum power.
                problem_size (int): The size of the problem.

            Raises:
                ValueError: when the given dataframe does not contain the required columns.

            Returns:
                float: The maximum power consumption of the solver.
            """
            if "solve_time" not in matching_df.columns:
                raise ValueError(
                    "The given dataframe does not contain the column 'solve_time'"
                )
            machine_time = np.mean(matching_df["solve_time"].values)
            power_max = machine_parameters["cpu_power"][problem_size]
            energy_max = power_max * machine_time
            return energy_max

        return cpu_energy_max_callable

    def cuda_energy_max(self, machine_parameters: dict = None):
        """The wrapper function of calculating the maximum energy consumption of the
        solver simulating on a cuda machine.

        Args:
            machine_parameters (dict, optional): Parameters of the. Defaults to None.

        Raises:
            ValueError: when the given machine parameters are not valid.
            ValueError: when the given dataframe does not contain the required columns.

        Returns:
            Callable: A callable function that takes in a dataframe and problem size and
                returns the maximum energy consumption of the solver.
        """
        if machine_parameters is None:
            machine_parameters = self._default_cuda_machine_parameters
        else:
            if "gpu_power" not in machine_parameters.keys():
                raise ValueError(
                    "The given machine parameters are not valid. "
                    "The dictionary must contain the key 'gpu_power'"
                )

        def cuda_energy_max_callable(matching_df: DataFrame, problem_size: int):
            """Calculate the maximum power consumption of the solver simulating on a
                cuda machine.

            Args:
                matching_df (DataFrame): The necessary data to calculate the maximum power.
                problem_size (int): The size of the problem.

            Raises:
                ValueError: when the given dataframe does not contain the required columns.

            Returns:
                float: The maximum power consumption of the solver.
            """
            if "solve_time" not in matching_df.columns:
                raise ValueError(
                    "The given dataframe does not contain the column 'solve_time'"
                )

            machine_time = np.mean(matching_df["solve_time"].values)
            power_max = machine_parameters["gpu_power"][problem_size]
            energy_max = power_max * machine_time
            return energy_max

        return cuda_energy_max_callable

    def energy_max(self, machine: MachineType, machine_parameters: dict = None):
        """Calculate the maximum power consumption of the solver simulating on a given machine.

        Args:
            machine (MachineType): The type of machine to calculate the maximum power consumption.
            machine_parameters (dict): Parameters of the machine. Defaults to None.

        Raises:
            ValueError: If the provided machine is not an instance of MachineType enum.
            ValueError: If the given machine type is not valid.
            ValueError: If there is a mismatch between the solver and the machine type.
        Returns:
            Callable: A callable function that calculates the maximum power consumption of
            the solver based on the given machine type.
        """
        solver_energy_methods = {
            MachineType.CPU: self.cpu_energy_max,
            MachineType.CUDA: self.cuda_energy_max,
            MachineType.DL_CCVM: self.optics_machine_energy
            if self.__class__.__name__ == "DLSolver"
            else None,
            MachineType.MF_CCVM: self.optics_machine_energy
            if self.__class__.__name__ == "MFSolver"
            else None,
            MachineType.FPGA: self.fpga_energy_max
            if self.__class__.__name__ == "LangevinSolver"
            else None,
        }

        if not isinstance(machine, MachineType):
            raise ValueError(
                "The provided machine must be an instance of MachineType enum."
            )

        if machine not in MachineType:
            raise ValueError(
                f"The given machine type is not valid. "
                f"The machine type must be one of {', '.join(m.value for m in MachineType)}"
            )

        energy_method = solver_energy_methods[machine]

        if not energy_method:
            raise ValueError(
                f"Mismatch between the solver and the machine type. "
                f"Provided machine type: {machine}, solver type: {self.__class__.__name__}"
            )

        return energy_method(machine_parameters)
