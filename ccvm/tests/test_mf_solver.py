from unittest import TestCase
from ccvm.solvers import MFSolver


class TestMFSolver(TestCase):
    def setUp(self):
        mf_solver = MFSolver(device="cpu", batch_size=1000, problem_category="boxqp")

    def test_set_parameter_key_with_valid_inputs(self):
        """Test parameter_key sets given parameters when inputs are valid"""
        # TODO: Implementation
        pass

    def test_set_parameter_key_with_invalid_keys(self):
        """Test that parameter_key throws an error when invalid parameter keys are passed"""
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
        """Test that tune successfully tunes the parameters with no errors"""
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
