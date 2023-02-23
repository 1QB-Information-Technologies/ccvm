import unittest
from unittest import TestCase
from boxqp.boxqp.problem_instance import ProblemInstance
import os
import torch

class TestProblemInstance(TestCase):
    def setUp(self):
        self.device = "cpu"
        self.instance_type = "basic"
        self.basepath = os.path.dirname(__file__)
        self.file_path = self.basepath + "/test_instances/test020-100-10.in"

    def test_constructor_with_valid_file_path(self):
        """Test the constructor when valid file path is provided"""

        problem_instance = ProblemInstance(
            device=self.device,
            instance_type=self.instance_type,
            file_path=self.file_path,
        )

        assert problem_instance.file_path == self.file_path

        assert problem_instance.device == self.device

        assert problem_instance.instance_type == self.instance_type

    def test_constructor_with_invalid_file_path(self):
        """Test the constructor when invalid file path is provided"""
        file_path = "/test_instances/test020-100-10.in"
        with self.assertRaises(FileNotFoundError):
            ProblemInstance(
                device=self.device,
                instance_type=self.instance_type,
                file_path=file_path,
            )

    def test_load_instance_cpu_and_basic_type(self):
        """Test load_instance when device is cpu and instance type is basic"""
        problem_instance = ProblemInstance(
                device=self.device,
                instance_type=self.instance_type,
                file_path=self.file_path,
            )
        #check all class variable type if function runs successfully 
        assert isinstance(problem_instance.optimal_sol, float)
        assert isinstance(problem_instance.sol_time_gb, float)
        assert torch.is_tensor(problem_instance.q)
        assert torch.is_tensor(problem_instance.c)

        error_message = "optimal_sol must be greater than 0"
        self.assertGreater(problem_instance.optimal_sol, 0, error_message)

        error_message = "sol_time_gb must be greater than 0"
        self.assertGreater(problem_instance.sol_time_gb, 0, error_message)
        

    def test_load_instance_cpu_and_tuning_type(self):
        """Test load_instance when device is cpu and instance type is tuning"""
        instance_type = "tuning"
        problem_instance = ProblemInstance(
                device=self.device,
                instance_type=instance_type,
                file_path=self.file_path,
            )

        #check all class variable type if function runs successfully 
        assert isinstance(problem_instance.optimal_sol, float)
        assert isinstance(problem_instance.sol_time_gb, float)
        assert torch.is_tensor(problem_instance.q)
        assert torch.is_tensor(problem_instance.c)

        error_message = "optimal_sol must be greater than 0"
        self.assertGreater(problem_instance.optimal_sol, 0, error_message)

        error_message = "sol_time_gb must be greater than 0"
        self.assertGreater(problem_instance.sol_time_gb, 0, error_message)

    def test_scale_coefs_one_time(self):
        """Test successfully scaling the instance's coefficients.
        """
        problem_instance = ProblemInstance(
                device=self.device,
                instance_type=self.instance_type,
                file_path=self.file_path,
            )

        confs = torch.FloatTensor(20, 20)
        problem_instance.scale_coefs(confs)
        assert torch.is_tensor(problem_instance.scaled_by)


    def test_compute_energy_times(self):
        """Test compute energy returns the right tensor
        """
        problem_instance = ProblemInstance(
                device=self.device,
                instance_type=self.instance_type,
                file_path=self.file_path,
            )

        confs = torch.FloatTensor(20, 20)
        energy = problem_instance.compute_energy(confs)
        assert torch.is_tensor(energy)
        

if __name__ == "__main__":
    unittest.main()
