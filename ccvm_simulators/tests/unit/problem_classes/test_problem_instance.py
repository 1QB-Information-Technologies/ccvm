import unittest
from unittest import TestCase
from ccvm_simulators.problem_classes.boxqp.problem_instance import ProblemInstance
import os
import torch


class TestProblemInstance(TestCase):
    def setUp(self):
        # Set default values
        self.device = "cpu"
        self.instance_type = "tuning"
        self.name = "test"
        self.file_delimiter = "\t"

        # Get the base path of the current file
        base_path = os.path.abspath(os.path.dirname(__file__))

        # Construct the path to the 'tests' folder
        test_folder = os.path.abspath(os.path.join(base_path, os.pardir, os.pardir))

        # Construct the path to the 'test_instances' folder
        self.test_instance_folder = os.path.join(test_folder, "data", "test_instances")

        # Set the file path
        self.file_path = os.path.join(self.test_instance_folder, "test020-100-10.in")

    def test_constructor_with_valid_file_path(self):
        """Test the constructor when valid file path is provided"""

        problem_instance = ProblemInstance(
            device=self.device,
            instance_type=self.instance_type,
            file_path=self.file_path,
            file_delimiter=self.file_delimiter,
            name=self.name,
        )

        assert problem_instance.file_path == self.file_path

        assert problem_instance.device == self.device

        assert problem_instance.instance_type == self.instance_type

        assert problem_instance.name == self.name

        assert problem_instance.file_delimiter == self.file_delimiter

    def test_name_parameter(self):
        """Test if constructor assigned a valid name"""

        problem_instance = ProblemInstance(
            device=self.device,
            instance_type=self.instance_type,
            file_path=self.file_path,
            file_delimiter=self.file_delimiter,
        )

        expected_name = self.file_path.split("/")[-1].split(".")[0]
        assert expected_name == problem_instance.name

    def test_valid_delimiter_assignment(self):
        """Test if constructor assigned a valid name"""

        problem_instance = ProblemInstance(
            device=self.device,
            instance_type=self.instance_type,
            file_path=self.file_path,
        )

        expected_file_delimiter = "\t"

        assert expected_file_delimiter == problem_instance.file_delimiter

        expected_name = self.file_path.split("/")[-1].split(".")[0]
        assert expected_name == problem_instance.name

    def test_constructor_with_invalid_file_path(self):
        """Test the constructor when invalid file path is provided"""
        invalid_file_path = "/test_instances/invalid.in"
        with self.assertRaises(FileNotFoundError):
            ProblemInstance(
                device=self.device,
                instance_type=self.instance_type,
                file_path=invalid_file_path,
            )

    def test_constructor_with_invalid_instance_type(self):
        """Test the constructor when invalid instance type"""
        invalid_instance_type = "dummy"
        with self.assertRaises(ValueError):
            ProblemInstance(
                device=self.device,
                instance_type=invalid_instance_type,
                file_path=self.file_path,
            )

    def test_load_instance_cpu(self):
        """Test load_instance when device is cpu and instance type is tuning"""
        instance_type = "tuning"
        problem_instance = ProblemInstance(
            device=self.device, instance_type=instance_type
        )

        problem_instance.load_instance(
            device=self.device,
            instance_type=instance_type,
            file_path=self.file_path,
            file_delimiter="\t",
        )

        expected_optimal_sol = 152.602291
        expected_best_sol = 147.960031

        # Check all class variable type if function runs successfully
        assert isinstance(problem_instance.optimal_sol, float)
        assert isinstance(problem_instance.best_sol, float)
        assert problem_instance.optimal_sol == expected_optimal_sol
        assert problem_instance.best_sol == expected_best_sol
        assert isinstance(problem_instance.sol_time_gb, float)
        assert isinstance(problem_instance.sol_time_bfgs, float)

        self.assertGreater(
            problem_instance.optimal_sol, 0, "optimal_sol must be greater than 0"
        )
        self.assertGreater(
            problem_instance.best_sol, 0, "best_sol must be greater than 0"
        )

        self.assertGreater(
            problem_instance.sol_time_gb, 0, "sol_time_gb must be greater than 0"
        )
        self.assertGreater(
            problem_instance.sol_time_bfgs, 0, "sol_time_bfgs must be greater than 0"
        )

    def test_scale_coefs_one_time(self):
        """Test successfully scaling the instance's coefficients."""
        file_path = self.test_instance_folder + "/test002.in"
        problem_instance = ProblemInstance(
            device=self.device,
            instance_type=self.instance_type,
            file_path=file_path,
        )

        scaling_factor = torch.FloatTensor([[1, 2], [10, 10]])
        expected_q_matrix = torch.FloatTensor([[41, 17.5], [2, 3]])
        expected_v_vector = torch.FloatTensor([[-31, -18.5], [-3.1, -3.7]])

        expected_scaled_by = scaling_factor * 1

        problem_instance.scale_coefs(scaling_factor)

        assert torch.equal(expected_scaled_by, problem_instance.scaled_by)

        assert torch.equal(expected_q_matrix, problem_instance.q_matrix)

        assert torch.equal(expected_v_vector, problem_instance.v_vector)

    def test_compute_energy_times(self):
        """Test compute energy returns the right tensor"""

        file_path = self.test_instance_folder + "/test002.in"

        problem_instance = ProblemInstance(
            device=self.device,
            instance_type=self.instance_type,
            file_path=file_path,
        )

        confs = torch.FloatTensor([[1, 2], [10, 10]])
        energy = problem_instance.compute_energy(confs)

        energy1 = (
            torch.einsum("bi, ij, bj -> b", confs, problem_instance.q_matrix, confs)
            * problem_instance.scaled_by
        )

        energy2 = (
            torch.einsum("bi, i -> b", confs, problem_instance.v_vector)
            * problem_instance.scaled_by
        )

        expected_result = 0.5 * energy1 + energy2

        assert torch.is_tensor(energy)
        assert torch.equal(expected_result, energy)

    def test_scale_coefs_multiple_times(self):
        file_path = self.test_instance_folder + "/test002.in"
        problem_instance = ProblemInstance(
            device=self.device,
            instance_type=self.instance_type,
            file_path=file_path,
        )

        scaling_factor = torch.FloatTensor([[10, 10], [10, 10]])
        frequency = 3

        for _ in range(frequency):
            problem_instance.scale_coefs(scaling_factor)

        expected_scaled_by = 1
        for _ in range(frequency):
            expected_scaled_by *= scaling_factor

        assert torch.equal(problem_instance.scaled_by, expected_scaled_by)

    def test_solution_bounds_valid(self):
        """Test that valid solution bounds can be set"""
        lower_bound = 1
        upper_bound = 10
        try:
            problem_instance = ProblemInstance(
                device=self.device,
                instance_type=self.instance_type,
                file_path=self.file_path,
                solution_bounds=(lower_bound, upper_bound),
            )
        except ValueError:
            self.fail("Solution bounds should be valid, unexpected value error")

    def test_solution_bounds_too_many_values(self):
        """Test that too many solution bounds values are rejected with the correct error message"""
        lower_bound = 1
        upper_bound = 10
        with self.assertRaises(ValueError) as context:
            ProblemInstance(
                device=self.device,
                instance_type=self.instance_type,
                file_path=self.file_path,
                solution_bounds=(lower_bound, upper_bound, 10),
            )
        self.assertEqual(
            str(context.exception),
            "solution_bounds must be a tuple of size 2, containing the minimum and maximum bounds (inclusive)",
        )

    def test_solution_bounds_too_few_values(self):
        """Test that too few solution bounds values are rejected with the correct error message"""
        lower_bound = 1
        with self.assertRaises(ValueError) as context:
            ProblemInstance(
                device=self.device,
                instance_type=self.instance_type,
                file_path=self.file_path,
                solution_bounds=(lower_bound,),
            )
        self.assertEqual(
            str(context.exception),
            "solution_bounds must be a tuple of size 2, containing the minimum and maximum bounds (inclusive)",
        )

    def test_solution_bounds_max_smaller_than_min(self):
        """Test that a solution bounds max value smaller than the min value is rejected with the correct error message"""
        lower_bound = 10
        upper_bound = 1
        with self.assertRaises(ValueError) as context:
            ProblemInstance(
                device=self.device,
                instance_type=self.instance_type,
                file_path=self.file_path,
                solution_bounds=(lower_bound, upper_bound),
            )
        self.assertEqual(
            str(context.exception),
            "Minimum solution bound must be less than maximum solution bound",
        )


if __name__ == "__main__":
    unittest.main()
