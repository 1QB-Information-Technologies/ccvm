from unittest import TestCase
from ccvm_simulators.solvers import LangevinSolver


class TestLangevinSolver(TestCase):
    def setUp(self):
        self.langevin_solver = LangevinSolver(
            device="cpu", batch_size=1000, problem_category="boxqp"
        )

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

    def test_fpga_machine_energy_default_parameters(self):
        """Test that _fpga_machine_energy returns a callable function with default parameters"""
        energy_callable = self.langevin_solver._fpga_machine_energy()
        self.assertTrue(callable(energy_callable))

    def test_fpga_machine_energy_custom_parameters(self):
        """Test that _fpga_machine_energy returns a callable function with custom parameters"""
        custom_parameters = {
            "fpga_power": {
                20: 15.18,
                30: 17.13,
                40: 17.45,
                50: 18.03,
                60: 18.22,
                70: 18.32,
            },
            "fpga_runtimes": {
                20: 133e-6,
                30: 265e-6,
                40: 327e-6,
                50: 437e-6,
                60: 511e-6,
                70: 662e-6,
            },
        }

        energy_callable = self.langevin_solver._fpga_machine_energy(
            machine_parameters=custom_parameters
        )
        self.assertTrue(callable(energy_callable))
