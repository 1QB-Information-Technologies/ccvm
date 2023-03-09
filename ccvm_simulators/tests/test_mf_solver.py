from unittest import TestCase, mock
from ccvm_simulators.solvers import MFSolver
import torch
import os
import csv
import random


class TestMFSolver(TestCase):
    def setUp(self):
        """Set up each test case with a new MFSolver instance and some valid parameters"""
        self.mf_solver = MFSolver(
            device="cpu", batch_size=1000, problem_category="boxqp"
        )
        self.batch_size = 1000
        self.problem_size = 2
        self.valid_parameters = {
            self.problem_size: {
                "pump": 2.5,
                "feedback_scale": 400,
                "j": 399,
                "S": 20.0,
                "lr": 0.0025,
                "iterations": 15000,
            }
        }

    def mock_calculate_grads(self, *args, **kwargs):
        """Mock the calculate_grads method to return dummy data"""
        return torch.zeros(self.batch_size, self.problem_size), torch.zeros(
            self.batch_size, self.problem_size
        )

    def mock_change_variables(self, *args, **kwargs):
        """Mock the change_variables method to return dummy data"""
        return torch.zeros(self.batch_size, self.problem_size)

    def mock_fit_to_constraints(self, *args, **kwargs):
        """Mock the fit_to_constraints method to return dummy data"""
        return torch.zeros(self.batch_size, self.problem_size)

    def test_set_parameter_key_with_valid_inputs(self):
        """Test parameter_key sets given parameters when inputs are valid"""
        self.mf_solver.parameter_key = self.valid_parameters
        assert self.mf_solver.parameter_key == self.valid_parameters

    def test_set_parameter_key_with_invalid_keys(self):
        """Test that parameter_key throws an error when invalid parameter keys are passed"""
        invalid_parameters = {
            2: {
                "pump": 2.5,
                "feedback_scale": 400,
                "j": 399,
                "S": 20.0,
                "lr": 0.0025,
                "iterations": 15000,
                "invalid_key": 1,
            }
        }
        with self.assertRaises(ValueError):
            self.mf_solver.parameter_key = invalid_parameters

    def test_method_selector_valid(self):
        """Test that method_selector set the correct methods when valid inputs are passed"""
        self.mf_solver._method_selector("boxqp")
        assert self.mf_solver.calculate_grads == self.mf_solver._calculate_grads_boxqp
        assert self.mf_solver.change_variables == self.mf_solver._change_variables_boxqp
        assert (
            self.mf_solver.fit_to_constraints
            == self.mf_solver._fit_to_constraints_boxqp
        )

    def test_method_selector_invalid(self):
        """Test that method_selector raises a ValueError when an invalid input is passed"""
        invalid_problem_category = "invalid_problem_category"
        with self.assertRaises(ValueError) as error:
            self.mf_solver._method_selector(invalid_problem_category)

        assert (
            str(error.exception)
            == f"The given problem category is not valid. Given category: {invalid_problem_category}"
        )

    def test_calculate_grads_boxqp_valid(self):
        """Test that calculate_grads returns correct data when valid parameters are passed"""
        batch_size = 3
        problem_size = 2

        mu = torch.zeros(batch_size, problem_size)
        mu_tilde = torch.zeros(batch_size, problem_size)
        sigma = torch.zeros(batch_size, problem_size)
        q_matrix = torch.ones(problem_size, problem_size)
        v_vector = torch.ones(problem_size)
        pump = self.valid_parameters[problem_size]["pump"]
        wiener_increment = torch.zeros(batch_size, problem_size)
        j = self.valid_parameters[problem_size]["j"]
        g = 0.1
        S = self.valid_parameters[problem_size]["S"]
        fs = self.valid_parameters[problem_size]["feedback_scale"]

        grads_mu, grads_sigma = self.mf_solver._calculate_grads_boxqp(
            mu=mu,
            mu_tilde=mu_tilde,
            sigma=sigma,
            q_matrix=q_matrix,
            v_vector=v_vector,
            pump=pump,
            wiener_increment=wiener_increment,
            j=j,
            g=g,
            S=S,
            fs=fs,
        )

        assert grads_mu.shape == torch.Size([batch_size, problem_size])
        assert grads_sigma.shape == torch.Size([batch_size, problem_size])

        # Expected values were calculated using the function. This check is just to make
        # sure that the function is not changed without updating the test.
        expected_grads_mu = torch.tensor(
            [[-20.0, -20.0], [-20.0, -20.0], [-20.0, -20.0]]
        )
        expected_grads_sigma = torch.tensor(
            [[200.5000, 200.5000], [200.5000, 200.5000], [200.5000, 200.5000]]
        )
        assert torch.equal(grads_mu, expected_grads_mu)
        assert torch.equal(grads_sigma, expected_grads_sigma)

    def test_change_variables_boxqp_valid(self):
        """Test that change_variables returns correct data when valid parameters are
        passed"""
        problem_variables = torch.tensor([[4.0, 4.0], [4.0, 4.0]])
        changed_variables = self.mf_solver._change_variables_boxqp(
            problem_variables=problem_variables, S=2
        )
        # Expected values were manually calculated
        expected_values = torch.tensor([[1.5, 1.5], [1.5, 1.5]])

        assert torch.equal(changed_variables, expected_values)

    def test_fit_to_constraints_boxqp_already_constrained(self):
        """
        Test that fit_to_constraints returns correct data when input parameter to clamp
        is already within the specified clamping range
        """
        mu_tilde = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        lower_clamp = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        upper_clamp = torch.tensor([[2.0, 2.0], [2.0, 2.0]])

        assert torch.equal(
            self.mf_solver._fit_to_constraints_boxqp(
                mu_tilde=mu_tilde, lower_clamp=lower_clamp, upper_clamp=upper_clamp
            ),
            mu_tilde,
        )

    def test_fit_to_constraints_boxqp_not_constrained(self):
        """
        Test that fit_to_constraints returns correct data when input parameter to clamp
        is not within the specified clamping range
        """
        mu_tilde = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        lower_clamp = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
        upper_clamp = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

        expected_values = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        assert torch.equal(
            self.mf_solver._fit_to_constraints_boxqp(
                mu_tilde=mu_tilde, lower_clamp=lower_clamp, upper_clamp=upper_clamp
            ),
            expected_values,
        )

    def test_fit_to_constraints_boxqp_partially_constrained(self):
        """
        Test that fit_to_constraints returns correct data when input parameter to clamp
        is partially within the specified clamping range
        """
        mu_tilde = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        lower_clamp = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
        upper_clamp = torch.tensor([[2.0, 2.0], [2.0, 2.0]])

        expected_values = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        assert torch.equal(
            self.mf_solver._fit_to_constraints_boxqp(
                mu_tilde=mu_tilde, lower_clamp=lower_clamp, upper_clamp=upper_clamp
            ),
            expected_values,
        )

    def test_append_samples_to_file(self):
        """Test that append_samples_to_file appends the correct data to the file"""

        # Throw an error if the file already exists
        test_file_path = "test_sample_file.txt"
        if os.path.exists(test_file_path):
            raise FileExistsError(
                "'test_sample_file.txt' exists; please remove or rename it before"
                " running this test"
            )

        # Create the file
        open(test_file_path, "w")

        try:
            # Set up the data to be written to the file
            mu_sample = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            sigma_sample = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

            # Call the function we are testing
            self.mf_solver._append_samples_to_file(
                mu_sample=mu_sample,
                sigma_sample=sigma_sample,
                evolution_file_object=open(test_file_path, "a"),
            )

            # Read the file and check that the data was written correctly
            with open(test_file_path, "r") as file:
                data = list(csv.reader(file, delimiter="\t"))
                assert data[0] == ["1.0", "2.0"]
                assert data[1] == ["3.0", "4.0"]
                assert data[2] == ["5.0", "6.0"]
                assert data[3] == ["7.0", "8.0"]
        finally:
            # Whether or not this test passed, delete the file
            os.remove(test_file_path)

    # TODO: More test cases should be created once function is implemented.
    def test_tune(self):
        """Test that tune successfully tunes the parameters with no errors"""
        # TODO: Implementation, once the tuning feature is implemented.
        pass

    def test_solve_success_minimal_inputs(self):
        """Test that solve will be successful when valid inputs are passed"""

        # Create a mock problem instance with arbitrary values
        instance = mock.MagicMock()
        instance.q_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        instance.v_vector = torch.tensor([[5.0, 6.0]])
        instance.problem_size = 2
        instance.compute_energy.return_value = torch.tensor([[7.0]])
        instance.optimal_sol = random.random()
        instance.device = "cpu"

        # Set up the solver with parameters and mock functions
        self.mf_solver.parameter_key = self.valid_parameters
        self.mf_solver.fit_to_constraints = self.mock_fit_to_constraints
        self.mf_solver.change_variables = self.mock_change_variables
        self.mf_solver.calculate_grads = self.mock_calculate_grads

        solution = self.mf_solver.solve(instance)

        # Check that the solution is correct
        assert solution.problem_size == instance.problem_size
        assert solution.batch_size == 1000
        assert solution.objective_values == instance.compute_energy.return_value
        assert (
            solution.iterations
            == self.valid_parameters[self.problem_size]["iterations"]
        )
        assert solution.objective_values == torch.tensor([[7.0]])
        assert solution.solve_time > 0.0
        # post-processing time should be 0.0 because the post processor defaults to None
        assert solution.pp_time == 0.0
        assert solution.optimal_value == instance.optimal_sol
        assert solution.device == self.mf_solver.device

        # Check that the correct variables exist in the solution
        assert "problem_variables" in solution.variables
        assert "mu" in solution.variables
        assert "sigma" in solution.variables
        # Check the values of the variables
        assert torch.is_tensor(solution.variables["problem_variables"])
        assert torch.is_tensor(solution.variables["mu"])
        assert torch.is_tensor(solution.variables["sigma"])

    def test_solve_device_mismatch(self):
        """Test that solve raises an error when the given device does not match the
        device in the problem instance"""

        # Create a mock problem instance with arbitrary values
        instance = mock.MagicMock()
        instance.q_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        instance.v_vector = torch.tensor([[5.0, 6.0]])
        instance.problem_size = 2
        instance.compute_energy.return_value = torch.tensor([[7.0]])

        instance.device = "cuda"

        self.mf_solver.device = "cpu"
        self.mf_solver.parameter_key = self.valid_parameters
        with self.assertRaises(ValueError) as error:
            self.mf_solver.solve(instance)

        self.assertEqual(
            str(error.exception),
            "The device type of the instance (cuda) and the solver (cpu) must match.",
        )
