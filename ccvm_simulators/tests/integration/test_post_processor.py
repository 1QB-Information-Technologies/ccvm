import logging
import os
import torch
from unittest import TestCase
from ...post_processor.adam import PostProcessorAdam
from ...post_processor.asgd import PostProcessorASGD
from ...post_processor.bfgs import PostProcessorBFGS
from ...post_processor.lbfgs import PostProcessorLBFGS
from ...post_processor.grad_descent import PostProcessorGradDescent

from ...problem_classes import ProblemInstance


class TestPostProcessor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger()
        cls.default_step_size = 0.1
        # cls.file_path = "examples/test_instances/test020-100-10.in"

        # Get the base path of the current file
        base_path = os.path.abspath(os.path.dirname(__file__))

        # Construct the path to the 'tests' folder
        test_folder = os.path.abspath(os.path.join(base_path, os.pardir))

        # Construct the path to the 'test_instances' folder
        test_instance_folder = os.path.join(test_folder, "data", "test_instances")

        # Set the file path
        cls.file_path = os.path.join(test_instance_folder, "test020-100-10.in")

        cls.problem_instance = ProblemInstance(file_path=cls.file_path)
        cls.v_vector = cls.problem_instance.v_vector
        cls.q_matrix = cls.problem_instance.q_matrix

    def setUp(self):
        self.N = 20
        self.M = 100
        self.c = torch.zeros(self.M, self.N)
        self.logger.info(f"Test {self._testMethodName} Started")

    def tearDown(self):
        self.logger.info(f"Test {self._testMethodName} Finished")

    def test_adam_postprocess_success(self):
        post_processor_adam = PostProcessorAdam()

        # Compute energy before postprocessing
        energy_before = self.problem_instance.compute_energy(self.c)

        output_tensor = post_processor_adam.postprocess(
            self.c, self.q_matrix, self.v_vector
        )

        # Compute energy after postprocessing
        energy_after = self.problem_instance.compute_energy(output_tensor)

        # Check if all elements of energy_after are smaller than or equal to the
        # corresponding elements in energy_before.
        self.assertTrue(torch.all(energy_after <= energy_before))

        # Check for NaN or Inf values in the tensor
        self.assertFalse(
            torch.isnan(output_tensor).any() or torch.isinf(output_tensor).any(),
            "Output contains NaN or Inf values",
        )

    def test_asgd_postprocess_success(self):
        post_processor_asgd = PostProcessorASGD()

        # Compute energy before postprocessing
        energy_before = self.problem_instance.compute_energy(self.c)

        output_tensor = post_processor_asgd.postprocess(
            self.c, self.q_matrix, self.v_vector
        )

        # Compute energy after postprocessing
        energy_after = self.problem_instance.compute_energy(output_tensor)

        # Check if all elements of energy_after are smaller than or equal to the
        # corresponding elements in energy_before.
        self.assertTrue(torch.all(energy_after <= energy_before))

        # Check for NaN or Inf values in the tensor
        self.assertFalse(
            torch.isnan(output_tensor).any() or torch.isinf(output_tensor).any(),
            "Output contains NaN or Inf values",
        )

    def test_bfgs_postprocess_success(self):
        post_processor_bfgs = PostProcessorBFGS()

        # Compute energy before postprocessing
        energy_before = self.problem_instance.compute_energy(self.c)

        output_tensor = post_processor_bfgs.postprocess(
            self.c, self.q_matrix, self.v_vector
        )

        # Compute energy after postprocessing
        energy_after = self.problem_instance.compute_energy(output_tensor)

        # Check if all elements of energy_after are smaller than or equal to the
        # corresponding elements in energy_before.
        self.assertTrue(torch.all(energy_after <= energy_before))
        # Check for NaN or Inf values in the tensor
        self.assertFalse(
            torch.isnan(output_tensor).any() or torch.isinf(output_tensor).any(),
            "Output contains NaN or Inf values",
        )

    def test_lbfgs_postprocess_success(self):
        post_processor_lbfgs = PostProcessorLBFGS()

        # Compute energy before postprocessing
        energy_before = self.problem_instance.compute_energy(self.c)

        output_tensor = post_processor_lbfgs.postprocess(
            self.c, self.q_matrix, self.v_vector
        )

        # Compute energy after postprocessing
        energy_after = self.problem_instance.compute_energy(output_tensor)

        # Check if all elements of energy_after are smaller than or equal to the
        # corresponding elements in energy_before.
        self.assertTrue(torch.all(energy_after <= energy_before))

        # Check for NaN or Inf values in the tensor
        self.assertFalse(
            torch.isnan(output_tensor).any() or torch.isinf(output_tensor).any(),
            "Output contains NaN or Inf values",
        )

    def test_grad_descent_postprocess_success(self):
        post_processor_grad_descent = PostProcessorGradDescent()

        # Compute energy before postprocessing
        energy_before = self.problem_instance.compute_energy(self.c)

        output_tensor = post_processor_grad_descent.postprocess(
            self.c, self.q_matrix, self.v_vector
        )

        # Compute energy after postprocessing
        energy_after = self.problem_instance.compute_energy(output_tensor)

        # Check if all elements of energy_after are smaller than or equal to the
        # corresponding elements in energy_before.
        self.assertTrue(torch.all(energy_after <= energy_before))

        # Check for NaN or Inf values in the tensor
        self.assertFalse(
            torch.isnan(output_tensor).any() or torch.isinf(output_tensor).any(),
            "Output contains NaN or Inf values",
        )
