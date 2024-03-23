import pandas as pd
import torch

from unittest import TestCase
from unittest.mock import MagicMock
from ccvm_simulators.solvers.ccvm_solver import (
    CCVMSolver,
    DeviceType,
    MachineType,
)
from ccvm_simulators.solvers.dl_solver import DLSolver
from ccvm_simulators.solvers.mf_solver import MFSolver
from ccvm_simulators.solvers.langevin_solver import LangevinSolver

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
        """Test that method_selector set the correct methods when valid inputs are
        passed."""
        self.solver._method_selector("boxqp")
        assert self.solver.calculate_grads == self.solver._calculate_grads_boxqp
        assert self.solver.change_variables == self.solver._change_variables_boxqp
        assert self.solver.fit_to_constraints == self.solver._fit_to_constraints_boxqp

    def test_method_selector_invalid(self):
        """Test that method_selector raises a ValueError when an invalid input is
        passed."""
        invalid_problem_category = "invalid_problem_category"
        with self.assertRaises(ValueError) as error:
            self.solver._method_selector(invalid_problem_category)

        assert (
            str(error.exception)
            == f"The given instance is not a valid problem category. Given category: {invalid_problem_category}"
        )


class TestCCVMSolverMachineEnergy(TestCase):

    def setUp(self):
        self.device = DeviceType.CPU_DEVICE.value
        self.solver = DummyConcreteSolver(device=self.device)
        self.dl_solver = DLSolver(device=self.device)
        self.mf_solver = MFSolver(device=self.device)
        self.langevin_solver = LangevinSolver(device=self.device)

    def test_validate_machine_energy_dataframe_columns_success(self):
        """Test that _validate_machine_energy_dataframe_columns does not raise an error
        when the dataframe contains the expected column names"""

        dataframe = pd.DataFrame(
            {"pp_time": [0.5, 0.6, 0.7], "iterations": [100, 200, 300]}
        )

        # No exception should be raised
        self.solver._validate_machine_energy_dataframe_columns(dataframe)

    def test_validate_machine_energy_dataframe_columns_missing_columns(self):
        """Test that _validate_machine_energy_dataframe_columns raises an error when the
        dataframe is missing columns"""
        # "iterations" column is missing
        dataframe = pd.DataFrame(
            {
                "pp_time": [0.5, 0.6, 0.7],
            }
        )

        with self.assertRaises(ValueError):
            self.solver._validate_machine_energy_dataframe_columns(dataframe)

    def test_extra_columns(self):
        """Test that _validate_machine_energy_dataframe_columns ignores extra columns in
        the dataframe."""

        dataframe = pd.DataFrame(
            {
                "pp_time": [0.5, 0.6, 0.7],
                "iterations": [100, 200, 300],
                "extra_column": [1, 2, 3],
            }
        )

        # No exception should be raised
        self.solver._validate_machine_energy_dataframe_columns(dataframe)

    def test_machine_energy_invalid_machine_type(self):
        """Test if ValueError is raised when machine type is invalid."""
        with self.assertRaises(ValueError):
            self.dl_solver.machine_energy("invalid_machine_type")

    def test_machine_energy_mismatch_solver_machine_type(self):
        """Test if ValueError is raised when machine type is not compatible with the solver."""
        with self.assertRaises(ValueError):
            self.langevin_solver.machine_energy(MachineType.DL_CCVM.value)

    def test_machine_energy_dl_solver_with_cpu_machine(self):
        """Test if machine_energy works correctly when machine type is cpu for
        DLSolver."""
        self.dl_solver._cpu_machine_energy = MagicMock(return_value=40.0)
        machine_energy = self.dl_solver.machine_energy(MachineType.CPU.value)
        self.assertEqual(machine_energy, 40.0)
        self.dl_solver._cpu_machine_energy.assert_called_once_with(None)

    def test_machine_energy_dl_solver_with_gpu_machine(self):
        """Test if machine_energy works correctly when machine type is gpu for DLSolver."""
        self.dl_solver._cuda_machine_energy = MagicMock(return_value=41.0)
        machine_energy = self.dl_solver.machine_energy(MachineType.GPU.value)
        self.assertEqual(machine_energy, 41.0)
        self.dl_solver._cuda_machine_energy.assert_called_once_with(None)

    def test_machine_energy_dl_solver_with_dl_machine(self):
        """Test if machine_energy works correctly when machine type is dl-ccvm for
        DLSolver."""
        self.dl_solver._optics_machine_energy = MagicMock(return_value=42.0)
        machine_parameters = {
            "laser_power": 1200e-6,
            "modulators_power": 10e-3,
            "squeezing_power": 180e-3,
            "electronics_power": 0.0,
            "amplifiers_power": 222.2e-3,
            "electronics_latency": 1e-9,
            "laser_clock": 10e-12,
            "postprocessing_power": {
                20: 4.96,
                30: 5.1,
                40: 4.95,
                50: 5.26,
                60: 5.11,
                70: 5.09,
            },
        }
        machine_energy = self.dl_solver.machine_energy(
            MachineType.DL_CCVM.value, machine_parameters
        )
        self.assertEqual(machine_energy, 42.0)
        self.dl_solver._optics_machine_energy.assert_called_once_with(
            machine_parameters
        )

    def test_machine_energy_mf_solver_with_cpu_machine(self):
        """Test if machine_energy works correctly when machine type is cpu for MFSolver."""
        self.mf_solver._cpu_machine_energy = MagicMock(return_value=40.0)
        machine_energy = self.mf_solver.machine_energy(MachineType.CPU.value)
        self.assertEqual(machine_energy, 40.0)
        self.mf_solver._cpu_machine_energy.assert_called_once_with(None)

    def test_machine_energy_mf_solver_with_gpu_machine(self):
        """Test if machine_energy works correctly when machine type is gpu for MFSolver."""
        self.mf_solver._cuda_machine_energy = MagicMock(return_value=41.0)
        machine_energy = self.mf_solver.machine_energy(MachineType.GPU.value)
        self.assertEqual(machine_energy, 41.0)
        self.mf_solver._cuda_machine_energy.assert_called_once_with(None)

    def test_machine_energy_mf_solver_with_mf_machine(self):
        """Test if machine_energy works correctly when machine type is mf-ccvm for
        MFSolver."""
        self.mf_solver._optics_machine_energy = MagicMock(return_value=43.0)
        machine_parameters = {
            "laser_clock": 100e-12,
            "FPGA_clock": 3.33e-9,
            "FPGA_fixed": 34,
            "FPGA_var_fac": 0.1,
            "FPGA_power": {
                20: 11.74,
                30: 14.97,
                40: 16.54,
                50: 18.25,
                60: 20.08,
                70: 22.01,
            },
            "buffer_time": 3.33e-9,
            "laser_power": 1000e-6,
            "postprocessing_power": {
                20: 4.96,
                30: 5.1,
                40: 4.95,
                50: 5.26,
                60: 5.11,
                70: 5.09,
            },
        }
        machine_energy = self.mf_solver.machine_energy(
            MachineType.MF_CCVM.value, machine_parameters
        )
        self.assertEqual(machine_energy, 43.0)
        self.mf_solver._optics_machine_energy.assert_called_once_with(
            machine_parameters
        )

    def test_machine_energy_langevin_solver_with_cpu_machine(self):
        """Test if machine_energy works correctly when machine type is cpu for Langevin
        Solver."""
        self.langevin_solver._cpu_machine_energy = MagicMock(return_value=40.0)
        machine_energy = self.langevin_solver.machine_energy(MachineType.CPU.value)
        self.assertEqual(machine_energy, 40.0)
        self.langevin_solver._cpu_machine_energy.assert_called_once_with(None)

    def test_machine_energy_mf_solver_with_gpu_machine(self):
        """Test if machine_energy works correctly when machine type is gpu for Langevin
        Solver."""
        self.langevin_solver._cuda_machine_energy = MagicMock(return_value=41.0)
        machine_energy = self.langevin_solver.machine_energy(MachineType.GPU.value)
        self.assertEqual(machine_energy, 41.0)
        self.langevin_solver._cuda_machine_energy.assert_called_once_with(None)

    def test_machine_energy_langevin_solver_with_fpga_machine(self):
        """Test if machine_energy works correctly when machine type is fpga for
        LangevinSolver."""
        self.langevin_solver._fpga_machine_energy = MagicMock(return_value=44.0)

        machine_parameters = {
            "fpga_power": {
                20: 17.18,
                30: 18.13,
                40: 18.45,
                50: 19.03,
                60: 19.22,
                70: 19.32,
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
        machine_energy = self.langevin_solver.machine_energy(
            MachineType.FPGA.value, machine_parameters
        )
        self.assertEqual(machine_energy, 44.0)
        self.langevin_solver._fpga_machine_energy.assert_called_once_with(
            machine_parameters
        )


class TestCCVMSolverMachineTime(TestCase):

    def setUp(self):
        self.device = DeviceType.CPU_DEVICE.value
        self.solver = DummyConcreteSolver(device=self.device)
        self.dl_solver = DLSolver(device=self.device)
        self.mf_solver = MFSolver(device=self.device)
        self.langevin_solver = LangevinSolver(device=self.device)

    def test_machine_time_invalid_machine_type(self):
        """Test if ValueError is raised when machine type is invalid."""
        with self.assertRaises(ValueError):
            self.dl_solver.machine_time("invalid_machine_type")

    def test_machine_time_mismatch_solver_machine_type(self):
        """Test if ValueError is raised when machine type is not compatible with the solver."""
        with self.assertRaises(ValueError):
            self.langevin_solver.machine_time(MachineType.DL_CCVM.value)

    def test_machine_time_dl_solver_with_cpu_machine(self):
        """Test if machine_time works correctly when machine type is cpu for
        DLSolver."""
        machine_parameters = {}
        cpu_callable = self.dl_solver.machine_time(
            machine=MachineType.CPU.value, machine_parameters=machine_parameters
        )

        # Check that the returned callable outputs the expected value
        dataframe = pd.DataFrame(data={"solve_time": [40.0, 20.0]})
        # Size not used by CPU version of this function, but test it can still be passed
        problem_size = 20
        self.assertEqual(
            cpu_callable(dataframe=dataframe, problem_size=problem_size), 30.0
        )

    def test_machine_time_dl_solver_with_gpu_machine(self):
        """Test if machine_time works correctly when machine type is gpu for
        DLSolver."""
        machine_parameters = {}
        gpu_callable = self.dl_solver.machine_time(
            machine=MachineType.GPU.value, machine_parameters=machine_parameters
        )

        # Check that the returned callable outputs the expected value
        dataframe = pd.DataFrame(data={"solve_time": [40.0, 20.0]})
        # Size not used by GPU version of this function, but test it can still be passed
        problem_size = 20
        self.assertEqual(
            gpu_callable(dataframe=dataframe, problem_size=problem_size), 30.0
        )

    def test_machine_time_mf_solver_with_cpu_machine(self):
        """Test if machine_time works correctly when machine type is cpu for
        MFSolver."""
        machine_parameters = {}
        cpu_callable = self.mf_solver.machine_time(
            machine=MachineType.CPU.value, machine_parameters=machine_parameters
        )

        # Check that the returned callable outputs the expected value
        dataframe = pd.DataFrame(data={"solve_time": [40.0, 20.0]})
        # Size not used by CPU version of this function, but test it can still be passed
        problem_size = 20
        self.assertEqual(
            cpu_callable(dataframe=dataframe, problem_size=problem_size), 30.0
        )

    def test_machine_time_dl_solver_with_dl_machine(self):
        """Test if machine_time works correctly when machine type is DL_CCVM for
        DLSolver."""
        machine_parameters = {
            "laser_power": 10e-12,
            "modulators_power": 10e-3,
            "squeezing_power": 180e-3,
            "electronics_power": 0.0,
            "amplifiers_power": 222.2e-3,
            "electronics_latency": 1e-9,
            "laser_clock": 9,
            "postprocessing_power": {
                20: 4.96,
            },
        }

        dl_callable = self.dl_solver.machine_time(
            machine=MachineType.DL_CCVM.value, machine_parameters=machine_parameters
        )

        # Check that the returned callable outputs the expected value
        dataframe = pd.DataFrame(data={"iterations": [4, 2], "pp_time": [16.0, 10.0]})
        problem_size = 20
        # Expected value was calculated manually based on the machine parameters and dataframe
        self.assertEqual(
            dl_callable(dataframe=dataframe, problem_size=problem_size), 553.0
        )

    def test_machine_time_langevin_solver_with_cpu_machine(self):
        """Test if machine_time works correctly when machine type is cpu for
        LangevinSolver."""
        machine_parameters = {}
        cpu_callable = self.langevin_solver.machine_time(
            machine=MachineType.CPU.value, machine_parameters=machine_parameters
        )

        # Check that the returned callable outputs the expected value
        dataframe = pd.DataFrame(data={"solve_time": [40.0, 20.0]})
        # Size not used by CPU version of this function, but test it can still be passed
        problem_size = 20
        self.assertEqual(
            cpu_callable(dataframe=dataframe, problem_size=problem_size), 30.0
        )

    def test_machine_time_mf_solver_with_gpu_machine(self):
        """Test if machine_time works correctly when machine type is gpu for
        MFSolver."""
        machine_parameters = {}
        gpu_callable = self.mf_solver.machine_time(
            machine=MachineType.GPU.value, machine_parameters=machine_parameters
        )

        # Check that the returned callable outputs the expected value
        dataframe = pd.DataFrame(data={"solve_time": [40.0, 20.0]})
        # Size not used by GPU version of this function, but test it can still be passed
        problem_size = 20
        self.assertEqual(
            gpu_callable(dataframe=dataframe, problem_size=problem_size), 30.0
        )

    def test_machine_time_langevin_solver_with_gpu_machine(self):
        """Test if machine_time works correctly when machine type is gpu for
        LangevinSolver."""
        machine_parameters = {}
        gpu_callable = self.langevin_solver.machine_time(
            machine=MachineType.GPU.value, machine_parameters=machine_parameters
        )

        # Check that the returned callable outputs the expected value
        dataframe = pd.DataFrame(data={"solve_time": [40.0, 20.0]})
        # Size not used by GPU version of this function, but test it can still be passed
        problem_size = 20
        self.assertEqual(
            gpu_callable(dataframe=dataframe, problem_size=problem_size), 30.0
        )

    def test_machine_time_mf_solver_with_mf_machine(self):
        """Test if machine_time works correctly when machine type is MF_CCVM for
        MFSolver."""
        machine_parameters = {
            "laser_clock": 2,
            "FPGA_clock": 5,
            "FPGA_fixed": 7,
            "FPGA_var_fac": 9,
            "buffer_time": 15,
            "FPGA_power": {20: 15.74},
            "laser_power": 1000e-6,
            "postprocessing_power": {20: 4.87},
        }

        mf_callable = self.mf_solver.machine_time(
            machine=MachineType.MF_CCVM.value, machine_parameters=machine_parameters
        )

        # Check that the returned callable outputs the expected value
        dataframe = pd.DataFrame(data={"iterations": [4, 2], "pp_time": [16.0, 10.0]})
        problem_size = 20
        # Expected value was calculated manually based on the machine parameters and dataframe
        self.assertEqual(
            mf_callable(dataframe=dataframe, problem_size=problem_size), 2983.0
        )

    def test_machine_time_langevin_solver_with_fpga_machine(self):
        """Test if machine_time works correctly when machine type is FPGA for
        LangevinSolver."""
        machine_parameters = {
            "fpga_power": {20: 17.18},
            "fpga_runtimes": {
                20: 7,
            },
        }

        langevin_callable = self.langevin_solver.machine_time(
            machine=MachineType.FPGA.value, machine_parameters=machine_parameters
        )

        # Check that the returned callable outputs the expected value
        dataframe = pd.DataFrame(data={"pp_time": [16.0, 10.0]})
        problem_size = 20
        # Expected value was calculated manually based on the machine parameters and dataframe
        self.assertEqual(
            langevin_callable(dataframe=dataframe, problem_size=problem_size), 20.0
        )
