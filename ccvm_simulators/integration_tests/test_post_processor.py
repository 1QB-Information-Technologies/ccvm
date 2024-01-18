import logging
import os
import torch
from unittest import TestCase
from ..post_processor.adam import PostProcessorAdam
from ..post_processor.asgd import PostProcessorASGD
from ..post_processor.bfgs import PostProcessorBFGS
from ..post_processor.lbfgs import PostProcessorLBFGS
from ..post_processor.trust_constr import PostProcessorTrustConstr
from ..post_processor.grad_descent import PostProcessorGradDescent

from ..problem_classes import ProblemInstance


class TestPostProcessor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger()
        cls.num_iter_main = 100
        cls.default_step_size = 0.1
        cls.N = 20
        cls.M = 100
        cls.c = torch.FloatTensor(cls.M, cls.N)
        # cls.v_vector = torch.randint(-50, 50, [cls.N])
        # cls.q_matrix_asym = torch.randint(-50, 50, [cls.N, cls.N])
        # cls.q_matrix = (
        #     torch.triu(cls.q_matrix_asym) + torch.triu(cls.q_matrix_asym, diagonal=1).T
        # )
        cls.file_path = "examples/test_instances/test020-100-10.in"
        cls.problem_instance = ProblemInstance(file_path=cls.file_path)

        cls.v_vector = cls.problem_instance.v_vector
        cls.q_matrix = cls.problem_instance.q_matrix

    def setUp(self):
        self.logger.info(f"Test {self._testMethodName} Started")

    def tearDown(self):
        self.logger.info(f"Test {self._testMethodName} Finished")

    def test_adam_postprocess_success(self):
        post_processor_adam = PostProcessorAdam()
        scaled_by = self.problem_instance.scaled_by
        print(f"scaled_by: {scaled_by}")
        # Compute energy before postprocessing
        energy_before = self.problem_instance.compute_energy(self.c)
        print(f"energy_before: {energy_before}")
        output_tensor = post_processor_adam.postprocess(
            self.c, self.q_matrix, self.v_vector
        )
        print(f"output_tensor: {output_tensor}")
        # Compute energy after postprocessing
        energy_after = self.problem_instance.compute_energy(output_tensor)
        print(f"energy_after: {energy_after}")

        # Check if all elements of energy_after are greater than or equal to the
        # corresponding elements in energy_before.
        self.assertTrue(torch.all(energy_after >= energy_before))

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

        # Check if all elements of energy_after are greater than or equal to the
        # corresponding elements in energy_before.
        self.assertTrue(torch.all(energy_after >= energy_before))

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

        # Check if all elements of energy_after are greater than or equal to the
        # corresponding elements in energy_before.
        self.assertTrue(torch.all(energy_after >= energy_before))
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

        # Check if all elements of energy_after are greater than or equal to the
        # corresponding elements in energy_before.
        self.assertTrue(torch.all(energy_after >= energy_before))

        # Check for NaN or Inf values in the tensor
        self.assertFalse(
            torch.isnan(output_tensor).any() or torch.isinf(output_tensor).any(),
            "Output contains NaN or Inf values",
        )

    def test_trust_constr_postprocess_success(self):
        post_processor_grad_descent = PostProcessorTrustConstr()

        # Compute energy before postprocessing
        energy_before = self.problem_instance.compute_energy(self.c)

        output_tensor = post_processor_grad_descent.postprocess(
            self.c, self.q_matrix, self.v_vector
        )

        # Compute energy after postprocessing
        energy_after = self.problem_instance.compute_energy(output_tensor)

        # Check if all elements of energy_after are greater than or equal to the
        # corresponding elements in energy_before.
        self.assertTrue(torch.all(energy_after >= energy_before))

        # Check for NaN or Inf values in the tensor
        self.assertFalse(
            torch.isnan(output_tensor).any() or torch.isinf(output_tensor).any(),
            "Output contains NaN or Inf values",
        )

    def test_grad_descent_postprocess_success(self):
        post_processor_grad_descent = PostProcessorGradDescent(self.num_iter_main)

        # Compute energy before postprocessing
        energy_before = self.problem_instance.compute_energy(self.c)

        output_tensor = post_processor_grad_descent.postprocess(
            self.c, self.q_matrix, self.v_vector
        )

        # Compute energy after postprocessing
        energy_after = self.problem_instance.compute_energy(output_tensor)

        # Check if all elements of energy_after are greater than or equal to the
        # corresponding elements in energy_before.
        self.assertTrue(torch.all(energy_after >= energy_before))

        # Check for NaN or Inf values in the tensor
        self.assertFalse(
            torch.isnan(output_tensor).any() or torch.isinf(output_tensor).any(),
            "Output contains NaN or Inf values",
        )
