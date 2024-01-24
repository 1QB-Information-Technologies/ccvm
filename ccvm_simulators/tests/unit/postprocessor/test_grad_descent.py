import logging
import torch
from unittest import TestCase
from ccvm_simulators.post_processor.grad_descent import PostProcessorGradDescent


class TestPostProcessorGradDescent(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger()
        cls.num_iter_main = 100
        cls.default_step_size = 0.1
        cls.post_processor = PostProcessorGradDescent(cls.num_iter_main)
        cls.N = 20
        cls.M = 100
        cls.c = torch.FloatTensor(cls.M, cls.N)
        cls.q_matrix = torch.FloatTensor(cls.N, cls.N)
        cls.v_vector = torch.FloatTensor(cls.N)

    def setUp(self):
        self.logger.info(f"Test {self._testMethodName} Started")

    def tearDown(self):
        self.logger.info(f"Test {self._testMethodName} Finished")

    def test_postprocess_default_values(self):
        """Test postprocess with default values for num_iter_pp and step_size"""
        output_tensor = self.post_processor.postprocess(
            self.c, self.q_matrix, self.v_vector
        )
        self.assertTrue(torch.is_tensor(output_tensor))
        self.assertEqual(output_tensor.size(), self.c.size())

    def test_postprocess_custom_num_iter_pp(self):
        """Test postprocess with custom num_iter_pp"""
        num_iter_pp = 10
        output_tensor = self.post_processor.postprocess(
            self.c, self.q_matrix, self.v_vector, num_iter_pp=num_iter_pp
        )
        self.assertTrue(torch.is_tensor(output_tensor))
        self.assertEqual(output_tensor.size(), self.c.size())

    def test_step_size_custom_value(self):
        """Test postprocess with custom step_size"""
        step_size = 0.05
        output_tensor = self.post_processor.postprocess(
            self.c, self.q_matrix, self.v_vector, step_size=step_size
        )
        self.assertTrue(torch.is_tensor(output_tensor))
        self.assertEqual(output_tensor.size(), self.c.size())

    def test_postprocess_invalid_c_parameter(self):
        """Test postprocess with invalid c parameter"""
        invalid_c = "dummy-c"
        with self.assertRaisesRegex(TypeError, "parameter c must be a tensor"):
            self.post_processor.postprocess(invalid_c, self.q_matrix, self.v_vector)

    def test_postprocess_invalid_qmat_parameter(self):
        """Test postprocess with invalid q_matrix parameter"""
        invalid_qmat = "dummy-qmat"
        with self.assertRaisesRegex(TypeError, "parameter q_matrix must be a tensor"):
            self.post_processor.postprocess(self.c, invalid_qmat, self.v_vector)

    def test_postprocess_invalid_v_vector_parameter(self):
        """Test postprocess with invalid v_vector parameter"""
        invalid_v_vector = "dummy-v_vector"
        with self.assertRaisesRegex(TypeError, "parameter v_vector must be a tensor"):
            self.post_processor.postprocess(self.c, self.q_matrix, invalid_v_vector)

    def test_postprocess_error_for_invalid_c_dimension(self):
        """Test postprocess with invalid c dimension"""
        incompatible_dimension = 9
        c = torch.FloatTensor(self.M, incompatible_dimension)
        with self.assertRaises(Exception):
            self.post_processor.postprocess(c, self.q_matrix, self.v_vector)

    def test_postprocess_error_for_invalid_v_vector_shape(self):
        """Test postprocess with invalid v_vector shape"""
        N = self.N
        v_vector = torch.FloatTensor(N, N)
        with self.assertRaises(Exception):
            self.post_processor.postprocess(self.c, self.q_matrix, v_vector)
