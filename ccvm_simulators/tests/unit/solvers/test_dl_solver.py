from unittest import TestCase
from ccvm_simulators.solvers import DLSolver


class TestDLSolver(TestCase):
    def setUp(self):
        self.dl_solver = DLSolver(device="cpu", batch_size=1000)

    def test_set_parameter_key_with_valid_inputs(self):
        """Test parameter_key sets given parameters when inputs are valid"""
        # TODO: Implementation
        pass

    def test_set_parameter_key_with_invalid_keys(self):
        """Test that parameter_key throws an error when invalid parameter keys are passed"""
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
        """Test that solve will be successful when valid inputs are passed"""
        # TODO: Implementation
        pass

    # TODO: More tests for solve may be required once it has implemented error handling.
    def test_solve_failed(self):
        """Test that solve return raise the correct error when invalid parameters are passed"""
        # TODO: Implementation
        pass

    def test_optics_machine_energy_default_parameters(self):
        """Test that _optics_machine_energy returns a callable function with default parameters"""
        energy_callable = self.dl_solver._optics_machine_energy()
        self.assertTrue(callable(energy_callable))

    def test_optics_machine_energy_custom_parameters(self):
        """Test that _optics_machine_energy returns a callable function with custom parameters"""
        custom_parameters = {
            "laser_power": 1000e-6,
            "modulators_power": 5e-3,
            "squeezing_power": 150e-3,
            "electronics_power": 0.0,
            "amplifiers_power": 200e-3,
            "electronics_latency": 1e-9,
            "laser_clock": 5e-12,
            "postprocessing_power": {
                20: 4.96,
                30: 5.1,
                40: 4.95,
                50: 5.26,
                60: 5.11,
                70: 5.09,
            },
        }
        energy_callable = self.dl_solver._optics_machine_energy(
            machine_parameters=custom_parameters
        )
        self.assertTrue(callable(energy_callable))
