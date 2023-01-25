import torch
from dataclasses import dataclass, field


class SolutionList:
    """Define the solutions class of a solve."""

    def __init__(self):
        """The constructor for SolutionList.
        :ivar solutions: a list of Solution
        :type solutions: list, default to empty
        """
        # Empty list to store solutions
        self.solutions = []

    def add_solution(
        self,
        problem_size,
        batch_size,
        instance_name,
        c_variables,
        objective_value,
        solve_time,
        pp_time,
        optimal_value,
        device="cpu",
    ):
        """Create a solution object with given inputs and append it to the solutions list.

        :param problem_size: The size of the problem solved
        :type problem_size: int
        :param batch_size: The batch size of the problem
        :type batch_size: int
        :param instance_name: The name of the problem instance
        :type instance_name: str
        :param c_variables: TODO _description_
        :type c_variables: torch.Tensor
        :param objective_value: The optimal solution of the problem
        :type objective_value: torch.Tensor
        :param solve_time: time to solve the problem
        :type solve_time: float
        :param pp_time: time to post-process the problem
        :type pp_time: float
        :param optimal_value: The optimal solution of the problem
        :type optimal_value: float
        :param device: Device to use, one of: "cpu" or "cuda"
        :type device: string, optional
        """
        # Check that the values that came from the solver are on the specified device
        # Otherwise, move them to the specified device
        if c_variables.get_device() != torch.device(device):
            c_variables = c_variables.to(device)
        if objective_value.get_device() != torch.device(device):
            objective_value = objective_value.to(device)

        solution = Solution(
            problem_size,
            batch_size,
            instance_name,
            c_variables,
            objective_value,
            solve_time,
            pp_time,
            optimal_value,
            device,
        )
        self.solutions.append(solution)


@dataclass
class Solution:
    """Similar to __init__() method of a normal class.
    :param problem_size: The size of the problem solved
    :type problem_size: int
    :param batch_size: The batch size of the problem
    :type batch_size: int
    :param instance_name: The name of the problem instance
    :type instance_name: str
    :param c_variables: the values of the problem variables found by the solver
    :type c_variables: torch.Tensor
    :param objective_value: The optimal solution of the problem
    :type objective_value: torch.Tensor
    :param solve_time: time to solve the problem
    :type solve_time: float
    :param pp_time: time to post-process the problem
    :type pp_time: float
    :param optimal_value: The optimal solution of the problem
    :type optimal_value: float
    :param device: Device to use, one of: "cpu" or "cuda"
    :type device: string
    :ivar solution_performance: a dictionary contains the following fields
        - optimal (float): Measurement of solutions that were optimal
        - one_percent (float): Measurement of solutions that were within 1% of optimal
        - two_percent (float):Measurement of solutions that were within 2% of optimal
        - three_percent (float):Measurement of solutions that were within 3% of optimal
        - four_percent (float):Measurement of solutions that were within 4% of optimal
        - five_percent (float):Measurement of solutions that were within 5% of optimal
        - ten_percent (float):Measurement of solutions that were within 10% of optimal
    :type solution_performance: dict, default to None
    """

    problem_size: int
    batch_size: int
    instance_name: str
    c_variables: torch.Tensor
    objective_value: torch.Tensor
    solve_time: float
    pp_time: float
    # When repr field set to False, the fields will be excluded from string representation
    optimal_value: float = field(repr=True)
    device: str = field(repr=True)
    solution_performance: dict = None

    def __post_init__(self):
        self.get_solution_stats()

    def get_solution_stats(self):

        """A method that calculates the measurement of solutions that were optimal, within 1%, 2%, 3%, 4%, 5% and 10% of optimal and update the solution."""
        objective_value = -self.objective_value
        device = self.device
        one_tensor = torch.ones(objective_value.size()).to(device)
        zero_tensor = torch.zeros(objective_value.size()).to(device)

        def fraction_below_threshold(gap_tensor, threshold):
            """Returns the fraction of the tensor's values that fall within the
            given threshold, rounded to 4 digits."""

            counter_tensor = torch.where(
                gap_tensor <= threshold, one_tensor, zero_tensor
            ).to(device)
            return round(counter_tensor.sum().item() / objective_value.size()[0], 4)

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
            (self.optimal_value - objective_value)
            * 100
            / torch.abs(objective_value).to(device)
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
