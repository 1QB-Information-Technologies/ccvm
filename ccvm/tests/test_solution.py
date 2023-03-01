import unittest
from ccvm.solution import Solution
import torch
import os


class TestSolution(unittest.TestCase):
    def setUp(self):
        self.problem_size = 10
        self.batch_size = 5
        self.instance_name = "test_instance"
        self.objective_value = torch.tensor((-1, -2, -3))
        self.solve_time = 2.0
        self.pp_time = 3.0
        self.optimal_value = 3.2
        self.device = "cpu"
        self.evolution_file = "test"
        self.iterations = 1
        self.variables = {"problem_variables": torch.tensor((10, 30, 50))}

    def test_save_file_invalid_tensor_name(self):
        solutions = Solution(
            self.problem_size,
            self.batch_size,
            self.instance_name,
            self.iterations,
            self.objective_value,
            self.solve_time,
            self.pp_time,
            self.optimal_value,
            self.variables,
            self.evolution_file,
            self.device,
        )

        with self.assertRaisesRegex(
            Exception, "Cannot find the test in the variables dictionary."
        ):
            solutions.save_tensor_to_file("test")

    def test_save_file_invalid_variable_type(self):
        variable = {"problem_variables": 10}
        solutions = Solution(
            self.problem_size,
            self.batch_size,
            self.instance_name,
            self.iterations,
            self.objective_value,
            self.solve_time,
            self.pp_time,
            self.optimal_value,
            variable,
            self.evolution_file,
            self.device,
        )

        tensor_name = "problem_variables"
        with self.assertRaisesRegex(
            Exception,
            f"A tensor object cannot be obtained by the given tensor_name: {tensor_name}",
        ):
            solutions.save_tensor_to_file("problem_variables")

    def test_save_file_valid_parameters(self):
        solutions = Solution(
            self.problem_size,
            self.batch_size,
            self.instance_name,
            self.iterations,
            self.objective_value,
            self.solve_time,
            self.pp_time,
            self.optimal_value,
            self.variables,
            self.evolution_file,
            self.device,
        )

        expected_path = os.path.dirname(__file__) + "/problem_variables.pt"

        assert not os.path.exists(expected_path)

        solutions.save_tensor_to_file("problem_variables", os.path.dirname(__file__))

        assert os.path.exists(expected_path)

        os.remove(expected_path)

    def test_solution_stats_objective_values_out_of_range(self):
        """Test the solution is updated with valid solution, even when the objective values 
        are unreasonable (we would expect them to be negative and lower in absolute value than 
        the optimal solution)
        """

        objective_value = torch.tensor((10, 30, 50))
        solutions = Solution(
            self.problem_size,
            self.batch_size,
            self.instance_name,
            self.iterations,
            objective_value,
            self.solve_time,
            self.pp_time,
            self.optimal_value,
            self.variables,
            self.evolution_file,
            self.device,
        )

        expected_solution_stats = {
            "optimal": 0,
            "one_percent": 0,
            "two_percent": 0,
            "three_percent": 0,
            "four_percent": 0,
            "five_percent": 0,
            "ten_percent": 0,
        }

        solutions.get_solution_stats()

        original_solution_stats = solutions.solution_performance
        assert original_solution_stats == expected_solution_stats

    def test_solution_stats_solutions_within_limit(self):
        """Test solution performance when objective values are close enough to the optimal solution 
            to yield results in the solution stats"""
        self.objective_value = torch.tensor((-0.007, -3.09, -3.199))
        solution = Solution(
            self.problem_size,
            self.batch_size,
            self.instance_name,
            self.iterations,
            self.objective_value,
            self.solve_time,
            self.pp_time,
            self.optimal_value,
            self.variables,
            self.evolution_file,
            self.device,
        )

        original_solution_stats = solution.solution_performance

        expected_solution_stats = {
            "optimal": 0.3333,
            "one_percent": 0.3333,
            "two_percent": 0.3333,
            "three_percent": 0.3333,
            "four_percent": 0.6667,
            "five_percent": 0.6667,
            "ten_percent": 0.6667,
        }

        assert original_solution_stats == expected_solution_stats

    def test_get_meta_dict(self):
        """Test solution performance for values parameters with no impact on solution stats"""

        solution = Solution(
            self.problem_size,
            self.batch_size,
            self.instance_name,
            self.iterations,
            self.objective_value,
            self.solve_time,
            self.pp_time,
            self.optimal_value,
            self.variables,
            self.evolution_file,
            self.device,
        )

        expected_result = {
            "problem_size": self.problem_size,
            "batch_size": self.batch_size,
            "instance_name": self.instance_name,
            "iterations": self.iterations,
            "solve_time": self.solve_time,
            "pp_time": self.pp_time,
            "optimal_value": self.optimal_value,
            "evolution_file": self.evolution_file,
            "solution_performance": {
                "optimal": 0.0,
                "one_percent": 0.0,
                "two_percent": 0.0,
                "three_percent": 0.0,
                "four_percent": 0.0,
                "five_percent": 0.0,
                "ten_percent": 0.3333,
            },
            "best_objective_value": 3,
        }

        actual_result = solution.get_metadata_dict()

        assert actual_result == expected_result


if __name__ == "__main__":
    unittest.main()
