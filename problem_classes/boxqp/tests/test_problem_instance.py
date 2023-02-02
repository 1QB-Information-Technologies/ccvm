from unittest import TestCase
from ..problem_instance import ProblemInstance


class TestProblemInstance(TestCase):
    def setUp(self):
        self.problem_instance = ProblemInstance()

    def test_constructor_with_valid_file_path(self):
        """Test the constructor when valid file path is provided"""
        # TODO: Implementation
        pass

    def test_constructor_with_invalid_file_path(self):
        """Test the constructor when invalid file path is provided"""
        # TODO: Implementation
        pass

    def test_constructor_when_no_file_path(self):
        """Test the constructor when no file path is provided"""
        # TODO: Implementation
        pass

    def test_load_instance_cpu_and_basic_type(self):
        """Test load_instance when device is cpu and instance type is basic"""
        # TODO: Implementation
        pass

    def test_load_instance_cpu_and_tuning_type(self):
        """Test load_instance when device is cpu and instance type is tuning"""
        # TODO: Implementation
        pass

    def test_load_instance_cuda_and_basic_type(self):
        """Test load_instance when device is cuda and instance type is basic"""
        # TODO: Implementation
        pass

    def test_load_instance_cuda_and_tuning_type(self):
        """Test load_instance when device is cuda and instance type is tuning"""
        # TODO: Implementation
        pass

    def test_load_instance_with_none_file_path(self):
        """Test load_instance when file_path is default value None"""
        # TODO: Implementation
        pass

    def test_load_instance_with_invalid_file_path(self):
        """Test load_instance when an invalid file path is provided"""
        # TODO: Implementation
        pass

    def test_scale_coefs_one_time(self):
        """Test successfully scaling the instance's coefficients."""
        # TODO: Implementation
        pass

    def test_scale_coefs_multiple_times(self):
        """Test successfully scaling the instance's coefficients consecutively.
        The resulting scaling factor should be the product of the scaling factors used.
        """
        # TODO: Implementation
        pass
