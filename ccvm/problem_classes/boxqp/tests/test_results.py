from unittest import TestCase
from unittest.mock import patch
from boxqp.results import Results, Result
import torch


class TestResult(TestCase):
    def setUp(self):
        self.problem_size = 10
        self.batch_size = 5
        self.instance_name = "test_instance"
        self.c_variables = torch.tensor((2, 4, 6))
        self.objective_value = torch.tensor((1, 3, 5))
        self.solve_time = self.pp_time = 2.0
        self.optimal_value = 3.2
        self.device = "cpu"

    def test_missing_required_inputs_error(self):
        """Test result constructor without providing objective_value"""
        with self.assertRaises(TypeError):
            Result(
                self.problem_size,
                self.batch_size,
                self.instance_name,
                self.c_variables,
                self.solve_time,
                self.pp_time,
                self.optimal_value,
                self.device,
            )

    @patch.object(Result, "get_result_stats")
    def test_initialize_result_success(self, mock_get_result_stats):
        """Test result constructor with valid inputs"""
        result = Result(
            self.problem_size,
            self.batch_size,
            self.instance_name,
            self.c_variables,
            self.objective_value,
            self.solve_time,
            self.pp_time,
            self.optimal_value,
            self.device,
        )
        mock_get_result_stats.assert_called_once()
        self.assertEqual(result.problem_size, self.problem_size)
        self.assertEqual(result.batch_size, self.batch_size)
        self.assertEqual(result.instance_name, self.instance_name)
        self.assertTrue(torch.equal(result.c_variables, self.c_variables))
        self.assertTrue(torch.equal(result.objective_value, self.objective_value))
        self.assertEqual(result.solve_time, self.solve_time)
        self.assertEqual(result.pp_time, self.pp_time)
        self.assertEqual(result.optimal_value, self.optimal_value)
        self.assertEqual(result.device, self.device)

    # TODO: may add more test cases when proper error handling is implemented
    def test_get_result_stats_success(self):
        """Test result stats can be calculated successfully"""
        # TODO: Implementation
        pass


class TestResults(TestCase):
    def setUp(self):
        self.problem_size = 10
        self.batch_size = 5
        self.instance_name = "test_instance"
        self.c_variables = torch.tensor((2, 4, 6))
        self.objective_value = torch.tensor((1, 3, 5))
        self.solve_time = self.pp_time = 2.0
        self.optimal_value = 3.2

        self.results = Results()

    # TODO: may add more test cases when proper error handling is implemented

    @patch.object(Result, "get_result_stats")
    def test_add_result_success(self, mock_get_result_stats):
        """Test the result can be successfully added to the results"""
        results_list = self.results.results
        self.assertEqual(len(results_list), 0)
        self.results.add_result(
            self.problem_size,
            self.batch_size,
            self.instance_name,
            self.c_variables,
            self.objective_value,
            self.solve_time,
            self.pp_time,
            self.optimal_value,
        )
        mock_get_result_stats.assert_called_once()
        self.assertEqual(len(results_list), 1)
