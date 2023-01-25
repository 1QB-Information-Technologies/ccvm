import torch
import enum


class DeviceType(enum.Enum):
    CPU_DEVICE = "cpu"
    CUDA_DEVICE = "cuda"


class InstanceType(enum.Enum):
    BASIC = "basic"
    TUNING = "tuning"
    TEST = "test"


# TODO: Revisit for a potential factory pattern
class ProblemInstance:
    """Defines a BoxQP problem instance.

    Attributes:
    :param device: Defines which GPU (or the CPU) to use.
    :type device: DeviceType, optional
    :param instance_type: TODO
    :type instance_type: ENUM, optional
    :param file_path: Path to file of problem instance.
    :type file_path: str, optional
    :param file_delimiter: The type of delimiter used in the file.
    :type file_delimiter: str, optional
    :param name: The name of the problem instance. If not given, defaults to the
    file name when an instance is loaded.
    :type name: str, optional
    :ivar N: TODO
    :vartype N: int, optional
    :ivar optimal_sol: TODO
    :vartype optimal_sol: float, optional
    :ivar optimality: TODO
    :vartype optimality: bool, optional
    :ivar sol_time_gb: TODO
    :vartype sol_time_gb: np, optional
    :ivar q: TODO
    :vartype q: Tensor, optional
    :ivar c: TODO
    :vartype c: Tensor, optional
    :ivar scaled_by: The amount the problem's terms have been scaled by, relative
    to the data that was loaded in. If the problem was not scaled, the tensor will hold the value 1.
    :vartype scaled_by: Tensor, optional
    """

    def __init__(
        self,
        device="cpu",
        instance_type="basic",
        file_path=None,
        file_delimiter="\t",
        name=None,
    ):
        """Constructor."""
        self.N = None
        self.optimal_sol = None
        self.optimality = None
        self.sol_time_gb = None
        self.q = None
        self.c = None
        self.scaled_by = 1
        self.device = device
        self.instance_type = instance_type
        self._custom_name = False
        self.file_delimiter = file_delimiter
        if name:
            self.name = name
            self._custom_name = True
        if file_path:
            self.file_path = file_path
            self.load_instance(
                device=device, instance_type=instance_type, file_path=file_path
            )

    def load_instance(
        self, device="cpu", instance_type="basic", file_path=None, file_delimiter=None
    ):
        """Loads in a box constraint problem from a file.

        :param device: Device to use, one of: "cpu" or "cuda"
        :type device: str, optional
        :param instance_type: Type of instance
        :type instance_type: str, optional
        :param file_path: Path to instance file
        :type file_path: str
        :param file_delimiter: Delimiter used in the instance file. If not specified,
        the file_delimiter value assigned at instance initialization will be used.
        :type file_delimiter: str, optional
        """
        rval_q = None
        rval_c = None
        N = None

        # Raise an exception if the file path was neither given as a load_instance
        # parameter nor upon initialization
        if not file_path and not self.file_path:
            raise Exception("No file path specified, cannot load instance.")

        # Update the file path if it was given as a load_instance parameter
        if file_path:
            self.file_path = file_path
        file_path = self.file_path

        # Update the file delimiter if it was given as a load_instance parameter
        if file_delimiter:
            self.file_delimiter = file_delimiter
        file_delimiter = self.file_delimiter

        # Read in data from the instance file
        with open(file_path, "r") as stream:
            try:
                # Read metadata from the first line
                lines = stream.readlines()
                instance_info = lines[0].split("\n")[0].split("\t")
                if instance_type == "basic":
                    # Save only the number of variables from the metadata
                    N = int(instance_info[0])
                    (optimality, optimal_sol, sol_time_gb) = (None, None, None)

                else:
                    # Save all metadata from the file
                    N = int(instance_info[0])
                    optimal_sol = float(instance_info[1])
                    if instance_info[2].lower() == "true":
                        optimality = True
                    else:
                        optimality = False
                    sol_time_gb = float(instance_info[3])

                # Initialize the q and c matrices
                rval_q = torch.zeros((N, N), dtype=torch.float).to(device)
                rval_c = torch.zeros((N,), dtype=torch.float).to(device)

                # Read in the c matrix
                line_data_c = lines[1].split("\n")[0].split(file_delimiter)
                for idx in range(0, N):
                    rval_c[idx] = -torch.Tensor([float(line_data_c[idx])])

                # Read in the q matrix
                for idx, line in enumerate(lines[2:]):
                    line_data = line.split("\n")[0].split(file_delimiter)
                    for j, value in enumerate(line_data[:N]):
                        rval_q[idx, j] = -torch.Tensor([float(value)])
            except Exception as e:
                raise Exception("Error reading instance file: " + str(e))

        # set class variables
        self.device = device
        self.instance_type = instance_type
        self.N = N
        self.optimal_sol = optimal_sol
        self.optimality = optimality
        self.sol_time_gb = sol_time_gb
        self.q = rval_q
        self.c = rval_c
        self.scaled_by = 1

        # Set the name of the instance if the user has not set it
        if not self._custom_name:
            # Remove the file extension and path, then name the instance after the file
            self.name = file_path.split("/")[-1].split(".")[0]

    def scale_coefs(self, scaling_factor):
        """Divides the coefficients of the problem stored in this instance by the given factor.
        Note that consectutive calls to this function will stack, e.g. scaling the problem
        by 4 twice would have the same result as scaling the original problem by 8.

        :param scaling_factor: The amount by which the coefficients should be scaled.
        :type scaling_factor: Tensor
        """
        self.q = self.q / scaling_factor
        self.c = self.c / scaling_factor
        self.scaled_by *= scaling_factor
