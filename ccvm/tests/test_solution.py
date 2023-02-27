import unittest
from unittest import TestCase
from unittest.mock import patch
from ccvm.solution import Solution
import torch
import os

class TestResult(TestCase):
    def setUp(self):
        self.problem_size = 10
        self.batch_size = 5
        self.instance_name = "test_instance"
        self.c_variables = torch.tensor((2000, 4000, 6000))
        self.objective_value = torch.tensor((10, 30, 50))
        self.solve_time = self.pp_time = 2.0
        self.pp_time = 3.0
        self.optimal_value = 3.2
        self.device = "cpu"
        self.variables = {"problem_variables": torch.tensor((10, 30, 50))}

    def test_save_file_invalid_tensor_name(self):
        solutions = Solution(
            self.problem_size,
            self.batch_size,
            self.instance_name,
            self.c_variables,
            self.objective_value,
            self.solve_time,
            self.pp_time,
            self.optimal_value,
            self.variables,
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
            self.c_variables,
            self.objective_value,
            self.solve_time,
            self.pp_time,
            self.optimal_value,
            variable,
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
            self.c_variables,
            self.objective_value,
            self.solve_time,
            self.pp_time,
            self.optimal_value,
            self.variables,
            self.device,
        )

        solutions.save_tensor_to_file("problem_variables",os.path.dirname(__file__))

        expected_path = os.path.dirname(__file__) + "/problem_variables.pt"

        assert os.path.exists(expected_path)

        os.remove(expected_path)

    
    def test_solution_stats_no_solution_within_limit(self):
        """Test the solution list is updated with valid solution"""
        solutions = Solution(
            self.problem_size,
            self.batch_size,
            self.instance_name,
            self.c_variables,
            self.objective_value,
            self.solve_time,
            self.pp_time,
            self.optimal_value,
            self.variables,
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
        """Test solution performance for values parameters with no impact on solution stats"""
        self.objective_value = torch.tensor((-2, -4, -5))
        solution = Solution(
            self.problem_size,
            self.batch_size,
            self.instance_name,
            self.c_variables,
            self.objective_value,
            self.solve_time,
            self.pp_time,
            self.optimal_value,
            self.variables,
            self.device,
        )

        solution.get_solution_stats()

        original_solution_stats = solution.solution_performance

        expected_solution_stats = {
            "optimal": 0.6667,
            "one_percent": 0.6667,
            "two_percent": 0.6667,
            "three_percent": 0.6667,
            "four_percent": 0.6667,
            "five_percent": 0.6667,
            "ten_percent": 0.6667,
        }

        assert original_solution_stats == expected_solution_stats


if __name__ == "__main__":
    unittest.main()
