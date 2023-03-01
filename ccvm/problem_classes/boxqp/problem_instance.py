import torch
import enum


class DeviceType(enum.Enum):
    """The devices that can be used by pytorch"""

    CPU_DEVICE = "cpu"
    CUDA_DEVICE = "cuda"


class InstanceType(enum.Enum):
    """Enumerate instance types."""

    TUNING = "tuning"
    TEST = "test"

    

# TODO: Revisit for a potential factory pattern
class ProblemInstance:
    """Defines a BoxQP problem instance."""

    def __init__(
        self,
        device="cpu",
        instance_type="tuning",
        file_path=None,
        file_delimiter="\t",
        name=None,
    ):
        """Problem instance constructor.

        Args:
            device (str, optional): Defines which GPU (or the CPU) to
                use. Defaults to "cpu".
            instance_type (str, optional): The type of the instance.
                Defaults to "tuning".
            file_path (str, optional): Path to file of problem instance.
                Defaults to None.
            file_delimiter (str, optional): The type of delimiter used in the
                file. Defaults to "\t".
            name (str, optional): The name of the problem instance. If not
                given, defaults to the file name when an instance is loaded.

        Attributes:
            problem_size (int): instance size. Defaults to None.
            optimal_sol (float): the optimal solution to the problem. Defaults to None.
            optimality (bool): indicates whether the solution is
                optimal (True or False). Defaults to None.
            q_matrix (torch.tensor): Q matrix of the QP problem. Defaults to None.
            v_vector (torch.tensor): V vector of the QP problem. Defaults to None.
            scaled_by (float): scaling value of the coefficient. Defaults to 1.
        """
        self.problem_size = None
        self.optimal_sol = None
        self.optimality = None
        self.sol_time_gb = None
        self.q_matrix = None
        self.v_vector = None
        self.scaled_by = 1
        self.device = device
        self._custom_name = False
        self.file_delimiter = file_delimiter
        instance_values = set(item.value for item in InstanceType)
        if instance_type in instance_values:
            self.instance_type = instance_type
        else:
            raise ValueError("instance_type must be tuning or test")
        if name:
            self.name = name
            self._custom_name = True
        if file_path:
            self.file_path = file_path
            self.load_instance(
                device=device,
                instance_type=instance_type,
                file_path=file_path,
                file_delimiter=file_delimiter,
            )
        self.problem_category = "boxqp"

    def load_instance(
        self, device="cpu", instance_type="tuning", file_path=None, file_delimiter=None
    ):
        """Loads in a box constraint problem from a file.

        Args:
            device (str, optional): Device to use. Defaults to "cpu".
            instance_type (str, optional): The type of the instance.
                Defaults to "tuning".
            file_path (str, optional): Path to instance file. Defaults to None.
            file_delimiter (str, optional): Delimiter used in the instance
                file. If not specified, the file_delimiter value assigned at
                instance initialization will be used.

        Raises:
            Exception: File path is not specified.
            Exception: Error reading the instance file.
        """
        rval_q = None
        rval_v = None
        problem_size = None

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
                instance_info = lines[0].split("\n")[0].split(file_delimiter)

                # Save all metadata from the file
                problem_size = int(instance_info[0])
                optimal_sol = float(instance_info[1])
                if instance_info[2].lower() == "true":
                    optimality = True
                else:
                    optimality = False
                sol_time_gb = float(instance_info[3])

                # Initialize the q_matrix and v_vector matrices
                rval_q = torch.zeros(
                    (problem_size, problem_size), dtype=torch.float
                ).to(device)
                rval_v = torch.zeros((problem_size,), dtype=torch.float).to(device)

                # Read in the v_vector matrix
                line_data_v = lines[1].split("\n")[0].split(file_delimiter)
                for idx in range(0, problem_size):
                    rval_v[idx] = -torch.Tensor([float(line_data_v[idx])])
                # Read in the q_matrix matrix
                for idx, line in enumerate(lines[2:]):
                    line_data = line.split("\n")[0].split(file_delimiter)
                    for j, value in enumerate(line_data[:problem_size]):
                        rval_q[idx, j] = -torch.Tensor([float(value)])
            except Exception as e:
                raise Exception("Error reading instance file: " + str(e))

        # set class variables
        self.device = device
        self.instance_type = instance_type
        self.problem_size = problem_size
        self.optimal_sol = optimal_sol
        self.optimality = optimality
        self.sol_time_gb = sol_time_gb
        self.q_matrix = rval_q
        self.v_vector = rval_v
        self.scaled_by = 1

        # Set the name of the instance if the user has not set it
        if not self._custom_name:
            # Remove the file extension and path, then name the instance after the file
            self.name = file_path.split("/")[-1].split(".")[0]

    def compute_energy(self, confs):
        """Compute the objective value for the given BoxQP instance using
        the formula '0.5 xQx + Vx', where 'x' is the vector of variables.

        Args:
            confs (torch.Tensor): Configurations for which to compute energy

        Returns:
            torch.Tensor: Energy of configurations.
        """
        energy1 = (
            torch.einsum("bi, ij, bj -> b", confs, self.q_matrix, confs)
            * self.scaled_by
        )
        energy2 = torch.einsum("bi, i -> b", confs, self.v_vector) * self.scaled_by
        return 0.5 * energy1 + energy2

    def scale_coefs(self, scaling_factor):
        """Divides the coefficients of the problem stored in this instance by
        the given factor. Note that consecutive calls to this function will
        stack, e.g. scaling the problem by 4 twice would have the same result as
        scaling the original problem by 16.

        Args:
            scaling_factor (torch.Tensor): The amount by which the
            coefficients should be scaled.
        """
        self.q_matrix = self.q_matrix / scaling_factor
        self.v_vector = self.v_vector / scaling_factor
        self.scaled_by *= scaling_factor
