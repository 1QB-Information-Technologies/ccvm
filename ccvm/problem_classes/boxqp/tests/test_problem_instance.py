import unittest
from unittest import TestCase
from ccvm.problem_classes.boxqp.problem_instance import ProblemInstance
import os
import torch


class TestProblemInstance(TestCase):
    def setUp(self):
        self.device = "cpu"
        self.instance_type = "tuning"
        self.basepath = os.path.dirname(__file__)
        self.file_path = self.basepath + "/test_instances/test020-100-10.in"
        self.name = "test"
        self.file_delimiter = "\t"

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
        """Test if constructor assigned a valid name """
   
        problem_instance = ProblemInstance(
            device=self.device,
            instance_type=self.instance_type,
            file_path=self.file_path,
            file_delimiter=self.file_delimiter,
        )

        expected_name = self.file_path.split("/")[-1].split(".")[0]
        assert expected_name == problem_instance.name

    def test_valid_delimiter_assignment(self):
        """Test if constructor assigned a valid name """
   
        problem_instance = ProblemInstance(
            device=self.device,
            instance_type=self.instance_type,
            file_path=self.file_path,
        )

        expected_file_delimiter="\t"

        
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

        expected_optimal_sol = 799.560976

        # check all class variable type if function runs successfully
        assert isinstance(problem_instance.optimal_sol, float)
        assert problem_instance.optimal_sol == expected_optimal_sol
        assert isinstance(problem_instance.sol_time_gb, float)

        error_message = "optimal_sol must be greater than 0"
        self.assertGreater(problem_instance.optimal_sol, 0, error_message)

        error_message = "sol_time_gb must be greater than 0"
        self.assertGreater(problem_instance.sol_time_gb, 0, error_message)

    def test_scale_coefs_one_time(self):
        """Test successfully scaling the instance's coefficients."""
        file_path = self.basepath + "/test_instances/test002.in"
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

        file_path = self.basepath + "/test_instances/test002.in"

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

        file_path = self.basepath + "/test_instances/test002.in"
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


if __name__ == "__main__":
    unittest.main()
