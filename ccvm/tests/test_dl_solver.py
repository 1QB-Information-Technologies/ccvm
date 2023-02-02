from unittest import TestCase
from ccvm.solvers import DLSolver


class TestMFSolver(TestCase):
    def setUp(self):
        dl_solver = DLSolver(device="cpu", batch_size=1000, problem_category="boxqp")

    def test_validate_parameters_with_valid_inputs(self):
        """Test _validate_parameters return true when given valid parameters"""
        # TODO: Implementation
        pass

    def test_validate_parameters_with_invalid_keys(self):
        """Test that _validate_parameters return false when invalid parameters are passed"""
        # TODO: Implementation
        pass

    def test_compute_energy_valid(self):
        """Test that compute_energy return correct value when valid parameters are passed"""
        # TODO: Implementation
        pass

    # TODO: More tests for compute_energy may be required once it has implemented error handling.
    def test_compute_energy_invalid(self):
        """Test that compute_energy raises the correct error when invalid parameters are passed"""
        # TODO: Implementation
        pass

    # TODO: Depending on the implementation, more test cases should be created.
    def test_tune(self):
        """Test that _validate_parameters return false when invalid parameters are passed"""
        # TODO: Implementation
        pass

    def test_calculate_grads_valid(self):
        """Test that calculate_grads returns correct data when valid parameters are passed"""
        # TODO: Implementation
        pass

    # TODO: More tests for calculate_grads may be required once it has implemented error handling.
    def test_calculate_grads_invalid(self):
        """Test that calculate_grads return raise the correct error when invalid parameters are passed"""
        # TODO: Implementation
        pass

    # TODO: Depending on the implementation, more test cases should be created.
    def test_solve_success(self):
        """Test that solve will be success when valid inputs are passed"""
        # TODO: Implementation
        pass

    # TODO: More tests for solve may be required once it has implemented error handling.
    def test_solve_failed(self):
        """Test that solve return raise the correct error when invalid parameters are passed"""
        # TODO: Implementation
        pass
