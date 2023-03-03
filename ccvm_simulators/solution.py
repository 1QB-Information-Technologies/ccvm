import torch
import os
from dataclasses import dataclass, field, asdict


@dataclass
class Solution:
    """Define the solution class of a solve.

    Args:
        problem_size (int): The size of the problem solved.
        batch_size (int): The number of times the problem instance was solved
            simultaneously.
        instance_name (str): The name of the problem instance.
        objective_values (torch.Tensor): The objective values of the solutions
            found by the solver.
        iterations (int): The iteration number for this problem size.
        solve_time (float): Time to solve the problem.
        pp_time (float):  Time to post-process the problem.
        optimal_value (float): The optimal objective value for the given problem instance.
        variables (dict): A dict object to store solution related variables.
            Some fields might be solver-dependent, such as mu, sigma and s.
            - problem_variables (torch.Tensor): The values of the problem
                variables found by the solver.
        evolution_file (str, optional): The filename of the evolution file, if it exists.
        device (str, optional): Device to use, one of: "cpu" or "cuda".
            Defaults to "cpu".

    Attributes:
        solution_performance (dict): A dictionary contains the following fields
            - optimal (float): The fraction of the solutions that were within the 0.1% of optimal value.
            - one_percent (float): The fraction of the solutions that were within the 1% of optimal value.
            - two_percent (float): The fraction of the solutions that were within the 2% of optimal value.
            - three_percent (float): The fraction of the solutions that were within the 3% of optimal value.
            - four_percent (float): The fraction of the solutions that were within the 4% of optimal value.
            - five_percent (float): The fraction of the solutions that were within the 5% of optimal value.
            - ten_percent (float): The fraction of the solutions that were within the 10% of optimal value.
            Defaults to None.
        best_objective_value (float): The best objective value found by the solver.
    """

    # When repr field set to False, the fields will be excluded from string
    # representation. Given tensor fileds are usually big, we set the repr
    # fields default to False to exclude tensor objects in metadata.
    problem_size: int
    batch_size: int
    instance_name: str
    iterations: int
    objective_values: torch.Tensor = field(repr=False)
    solve_time: float
    pp_time: float
    optimal_value: float
    variables: dict = field(repr=False)
    evolution_file: str = None
    device: str = field(default="cpu", repr=False)
    solution_performance: dict = None
    best_objective_value: float = None

    def __post_init__(self):
        """Runs automatically after Solution initialization."""

        # Check that the values that came from the solver are on the specified device
        # Otherwise, move them to the specified device
        device = self.device
        objective_values = self.objective_values
        for key, value in self.variables.items():
            if torch.is_tensor(value) and value.device != torch.device(device):
                self.variables[key] = value.to(device)

        if torch.is_tensor(
            objective_values
        ) and objective_values.device != torch.device(device):
            self.objective_values = objective_values.to(device)

        # Find the best objective value
        self.best_objective_value = torch.max(-self.objective_values).item()

        # Calculate and update solution_performance
        self.get_solution_stats()

    def get_solution_stats(self):
        """A method that calculates the fraction of solutions that were optimal,
        within 1%, 2%, 3%, 4%, 5% and 10% of optimal value and update the
        solution."""
        objective_values = -self.objective_values
        device = self.device
        one_tensor = torch.ones(objective_values.size()).to(device)
        zero_tensor = torch.zeros(objective_values.size()).to(device)

        def fraction_below_threshold(gap_tensor, threshold):
            """Returns the fraction of the tensor's values that fall within the given threshold, rounded to 4 digits.

            Args:
                gap_tensor (torch.Tensor): The tensor of the percentage distances of the found
                    objective values from the optimal value of the problem objective function.
                threshold (float): The specified percentage gap

            Returns:
                torch.Tensor: the fraction of the solved instances where the found solution
                was within the speficied percentage gap from the optimal objective value.

            """

            counter_tensor = torch.where(
                gap_tensor <= threshold, one_tensor, zero_tensor
            ).to(device)
            return round(counter_tensor.sum().item() / objective_values.size()[0], 4)

        (
            optimal,
            one_percent,
            two_percent,
            three_percent,
            four_percent,
            five_percent,
            ten_percent,
        ) = (0, 0, 0, 0, 0, 0, 0)

        gap_tensor = (
            (self.optimal_value - objective_values)
            * 100
            / torch.abs(objective_values).to(device)
        )
        optimal = fraction_below_threshold(gap_tensor, 0.1)
        one_percent = fraction_below_threshold(gap_tensor, 1)
        two_percent = fraction_below_threshold(gap_tensor, 2)
        three_percent = fraction_below_threshold(gap_tensor, 3)
        four_percent = fraction_below_threshold(gap_tensor, 4)
        five_percent = fraction_below_threshold(gap_tensor, 5)
        ten_percent = fraction_below_threshold(gap_tensor, 10)

        self.solution_performance = {
            "optimal": optimal,
            "one_percent": one_percent,
            "two_percent": two_percent,
            "three_percent": three_percent,
            "four_percent": four_percent,
            "five_percent": five_percent,
            "ten_percent": ten_percent,
        }

    def get_metadata_dict(self) -> dict:
        """Return the metadata dictonary. Excluding the tensor fields in
        solution by filtering out the fields where repr sets to False.

        Returns:
            dict: metadata of the solution.
        """
        return {
            k: v for k, v in asdict(self).items() if self.__dataclass_fields__[k].repr
        }

    def save_tensor_to_file(self, tensor_name, file_dir=".", file_name=None):
        """Save the tensor that exists in the solution variables dictionary to a file.

        Args:
            tensor_name (str): The name (key) that identifies the tensor in the
                solution.variable dictionary.
            file_dir (str): The directory of the file. Default to current directory.
            file_name (str): The name of the file storing tensor. If not
                provided, defaults to the same name as tensor_name.

        Raises:
            Exception: Failed to create the folder path.
            Exception: Tensor_name not exists in Solution.
            Exception: A tensor object cannot be obtained by tensor_name.
        """
        # Get all the keys defined in the variable attribute
        keys = self.variables.keys()

        # If a customized solutions_dir is provided and not exists, create the
        # path
        try:
            if file_dir != "." and not os.path.isdir(file_dir):
                os.makedirs(file_dir)
                print("The folder to store doesn't exist yet. Creating: ", file_dir)
        except Exception as e:
            raise Exception(f"Failed to create the folder path: {e}")

        if tensor_name not in keys:
            raise Exception(
                f"Cannot find the {tensor_name} in the variables dictionary."
            )
        elif not file_name:
            file_name = tensor_name

        tensor_value = getattr(self, "variables")[tensor_name]
        if torch.is_tensor(tensor_value):
            torch.save(tensor_value, f"{file_dir}/{file_name}.pt")
            print("Successfully saved the tensor!")
        else:
            raise Exception(
                f"A tensor object cannot be obtained by the given tensor_name: {tensor_name}"
            )
