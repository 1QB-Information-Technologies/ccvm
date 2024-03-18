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
    GPU = "gpu"
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
            float: The recommended scaling factor to be use to scale the problem for
                this solver.
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

    ################################
    ### MACHINE ENERGY FUNCTIONS ###
    ################################

    def _validate_machine_energy_dataframe_columns(self, dataframe):
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

    def _cpu_machine_energy(self, machine_parameters: dict = None):
        """The wrapper function of calculating the average energy consumption of the
        solver simulating on a CPU machine.

        Args:
            machine_parameters (dict, optional): Parameters of the CPU. Defaults to None.

        Raises:
            ValueError: when the given machine parameters are not valid.
            ValueError: when the given dataframe does not contain the required columns.

        Returns:
            Callable: A callable function that takes in a dataframe and problem size and
                returns the average energy consumption of the solver.
        """
        if machine_parameters is None:
            machine_parameters = self._default_cpu_machine_parameters
        else:
            if "cpu_power" not in machine_parameters.keys():
                raise ValueError(
                    "The given machine parameters are not valid. "
                    "The dictionary must contain the key 'cpu_power'"
                )

        def _cpu_machine_energy_callable(dataframe: DataFrame, problem_size: int):
            """Calculate the average energy consumption of the solver simulating on a
            cpu machine.

            Args:
                dataframe (DataFrame): The necessary data to calculate the average
                    energy.
                problem_size (int): The size of the problem.

            Raises:
                ValueError: when the given dataframe does not contain the required
                    columns.

            Returns:
                float: The average energy consumption of the solver.
            """
            if "solve_time" not in dataframe.columns:
                raise ValueError(
                    "The given dataframe does not contain the column 'solve_time'"
                )
            machine_time = np.mean(dataframe["solve_time"].values)
            machine_power = machine_parameters["cpu_power"][problem_size]
            machine_energy = machine_power * machine_time
            return machine_energy

        return _cpu_machine_energy_callable

    def _cuda_machine_energy(self, machine_parameters: dict = None):
        """The wrapper function of calculating the average energy consumption of the
        solver simulating on system equipped with CUDA-capable GPUs.

        Args:
            machine_parameters (dict, optional): Parameters of the CUDA-capable GPUs.
            Defaults to None.

        Raises:
            ValueError: when the given machine parameters are not valid.
            ValueError: when the given dataframe does not contain the required columns.

        Returns:
            Callable: A callable function that takes in a dataframe and problem size and
                returns the average energy consumption of the solver.
        """
        if machine_parameters is None:
            machine_parameters = self._default_cuda_machine_parameters
        else:
            if "gpu_power" not in machine_parameters.keys():
                raise ValueError(
                    "The given machine parameters are not valid. "
                    "The dictionary must contain the key 'gpu_power'"
                )

        def _cuda_machine_energy_callable(dataframe: DataFrame, problem_size: int):
            """Calculate the average energy consumption of the solver simulating on a
            system equipped with CUDA-capable GPUs.

            Args:
                dataframe (DataFrame): The necessary data to calculate the average
                    energy.
                problem_size (int): The size of the problem.

            Raises:
                ValueError: when the given dataframe does not contain the required
                    columns.

            Returns:
                float: The average power consumption of the solver.
            """
            if "solve_time" not in dataframe.columns:
                raise ValueError(
                    "The given dataframe does not contain the column 'solve_time'"
                )

            machine_time = np.mean(dataframe["solve_time"].values)
            machine_power = machine_parameters["gpu_power"][problem_size]
            machine_energy = machine_power * machine_time
            return machine_energy

        return _cuda_machine_energy_callable

    def machine_energy(self, machine: str, machine_parameters: dict = None):
        """Calculates the average energy consumed by the specified hardware for a given
        problem size.

        Args:
            machine (str): The type of machine to calculate the average energy consumption.
            machine_parameters (dict): Parameters of the machine. Defaults to None.

        Raises:
            ValueError: If the given machine is not a valid machine type.
            ValueError: If there is a mismatch between the solver and the machine type.
        Returns:
            Callable: A callable function that calculates the average energy consumption
                of the solver based on the given machine type.
        """
        solver_energy_methods = {
            "cpu": self._cpu_machine_energy,
            "gpu": self._cuda_machine_energy,
            "dl-ccvm": (
                self._optics_machine_energy
                if self.__class__.__name__ == "DLSolver"
                else None
            ),
            "mf-ccvm": (
                self._optics_machine_energy
                if self.__class__.__name__ == "MFSolver"
                else None
            ),
            "fpga": (
                self._fpga_machine_energy
                if self.__class__.__name__ == "LangevinSolver"
                else None
            ),
        }

        if machine not in solver_energy_methods:
            raise ValueError(
                f"The given machine type is not valid. "
                f"The machine type must be one of {', '.join(solver_energy_methods.keys())}"
            )

        energy_method = solver_energy_methods[machine]

        if not energy_method:
            raise ValueError(
                f"Mismatch between the solver and the machine type. "
                f"Provided machine type: {machine}, solver type: {self.__class__.__name__}"
            )

        return energy_method(machine_parameters)

    ##############################
    ### MACHINE TIME FUNCTIONS ###
    ##############################

    def _cpu_gpu_machine_time(self, **_):
        """The wrapper function of calculating the average time taken by the solver during
        the simulation when using a CPU or a CUDA-capable GPU machine.

        Raises:
            ValueError: when the given dataframe does not contain the required columns.

        Returns:
            Callable: A callable function that takes in a dataframe and problem size and
                returns the average time taken by the solver.
        """

        def _cpu_gpu_machine_time_callable(dataframe: DataFrame, **_):
            """Calculate the average time taken by the solver during the simulation when
            using a CPU or a CUDA-capable GPU machine.

            Args:
                dataframe (DataFrame): The necessary data to calculate the average
                    time spent during the simulation.
                problem_size (int): The size of the problem.

            Raises:
                ValueError: when the given dataframe does not contain the required
                    columns.

            Returns:
                float: The average time taken by the solver during simulation of a single
                    instance.
            """
            if "solve_time" not in dataframe.columns:
                raise ValueError(
                    "The given dataframe does not contain the column 'solve_time'"
                )
            machine_time = np.mean(dataframe["solve_time"].values)
            return machine_time

        return _cpu_gpu_machine_time_callable

    def machine_time(self, machine: str, machine_parameters: dict = None):
        """Calculates the average time spent during the simulation by the specified hardware
        for a given problem size.

        Args:
            machine (str): The type of machine for which to calculate the average time for
                simulating a single instance.
            machine_parameters (dict): Parameters of the machine. Defaults to None.

        Raises:
            ValueError: If the given machine is not a valid machine type.
            ValueError: If there is a mismatch between the solver and the machine type.
        Returns:
            Callable: A callable function that calculates the average time taken by the
                solver during simulation of a single instance on the given machine type.
        """
        solver_time_methods = {
            "cpu": self._cpu_gpu_machine_time,
            "gpu": self._cpu_gpu_machine_time,
            "dl-ccvm": (
                self._optics_machine_time
                if self.__class__.__name__ == "DLSolver"
                else None
            ),
            # "mf-ccvm": self._optics_machine_time
            # if self.__class__.__name__ == "MFSolver"
            # else None,
            # "fpga": self._fpga_machine_time
            # if self.__class__.__name__ == "LangevinSolver"
            # else None,
        }

        if machine not in solver_time_methods:
            raise ValueError(
                f"The given machine type is not valid. "
                f"The machine type must be one of {', '.join(solver_time_methods.keys())}"
            )

        time_method = solver_time_methods[machine]

        if not time_method:
            raise ValueError(
                f"Mismatch between the solver and the machine type. "
                f"Provided machine type: {machine}, solver type: {self.__class__.__name__}"
            )

        return time_method(machine_parameters=machine_parameters)
