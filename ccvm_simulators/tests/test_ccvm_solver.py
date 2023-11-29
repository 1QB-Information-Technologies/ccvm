from unittest import TestCase
from ccvm_simulators.solvers.ccvm_solver import CCVMSolver, DeviceType
import torch

DUMMY_SCALING_MULTIPLIER = 0.1


class DummyConcreteSolver(CCVMSolver):
    # This dummy concrete solver class is used for testing the abstract class
    def __init__(self, device):
        super().__init__(device)
        self._scaling_multiplier = DUMMY_SCALING_MULTIPLIER

    # Add the dummy implementation of abstract methods. These methods won't be
    # tested, but need to exist in this subclass.
    def _validate_parameters(self, parameters):
        print("dummy _validate_parameters function")

    def tune(self):
        print("dummy tune function")

    def _calculate_grads_boxqp(self):
        print("dummy _calculate_grads_boxqp function")

    def _change_variables_boxqp(self):
        print("dummy _change_variables_boxqp function")

    def _fit_to_constraints_boxqp(self):
        print("dummy _fit_to_constraints_boxqp function")

    def _calculate_drift_boxqp(self):
        print("dummy _calculate_drift_boxqp function")

    def _solve(self):
        print("dummy solve function")

    def _solve_adam(self):
        print("dummy _solve_adam function")


class TestCCVMSolver(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameter_key = {
            20: {
                "pump": 2.5,
                "feedback_scale": 400,
                "j": 399,
                "S": 20.0,
                "dt": 0.0025,
                "iterations": 15000,
            }
        }

    def setUp(self):
        self.device = DeviceType.CPU_DEVICE.value
        self.solver = DummyConcreteSolver(device=self.device)

    def test_constructor_with_invalid_device(self):
        """Test the CCVM solver constructor when pass in invalid device"""
        invalid_device = "invald_device"
        with self.assertRaises(ValueError) as context:
            solver = DummyConcreteSolver(device=invalid_device)

        self.assertTrue("Given device is not available" in str(context.exception))

    def test_is_tuned_property_default_value_correct(self):
        # The default value is False according to our init implmentation
        self.assertFalse(self.solver.is_tuned)

    def test_parameter_key_property_default_value_correct(self):
        # The default value is None according to our init implmentation
        self.assertIsNone(self.solver.parameter_key)

    def test_get_scaling_factor_success(self):
        problem_size = 20
        q_matrix = torch.rand(problem_size, problem_size)
        expected_value = (
            torch.sqrt(torch.sum(torch.abs(q_matrix))) * DUMMY_SCALING_MULTIPLIER
        )

        self.assertTrue(
            torch.eq(self.solver.get_scaling_factor(q_matrix), expected_value)
        )

    def test_method_selector_valid(self):
        """Test that method_selector set the correct methods when valid inputs are passed"""
        self.solver._method_selector("boxqp")
        assert self.solver.calculate_grads == self.solver._calculate_grads_boxqp
        assert self.solver.change_variables == self.solver._change_variables_boxqp
        assert self.solver.fit_to_constraints == self.solver._fit_to_constraints_boxqp

    def test_method_selector_invalid(self):
        """Test that method_selector raises a ValueError when an invalid input is passed"""
        invalid_problem_category = "invalid_problem_category"
        with self.assertRaises(ValueError) as error:
            self.solver._method_selector(invalid_problem_category)

        assert (
            str(error.exception)
            == f"The given instance is not a valid problem category. Given category: {invalid_problem_category}"
        )
