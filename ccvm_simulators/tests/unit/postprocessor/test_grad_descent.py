import logging
import torch
from unittest import TestCase
from ccvm_simulators.post_processor.grad_descent import PostProcessorGradDescent


class TestPostProcessorGradDescent(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger()
        cls.default_step_size = 0.1
        cls.post_processor = PostProcessorGradDescent()
        cls.N = 20
        cls.M = 100
        cls.c = torch.zeros(cls.M, cls.N)
        cls.q_matrix_asym = torch.randint(-50, 50, [cls.N, cls.N])
        cls.v_vector = torch.randint(-50, 50, [cls.N])
        cls.q_matrix = (
            torch.triu(cls.q_matrix_asym) + torch.triu(cls.q_matrix_asym, diagonal=1).T
        )

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

    def test_postprocess_custom_upper_lower_clamp(self):
        # Test with custom values for lower_clamp and upper_clamp
        lower_clamp = -1.0
        upper_clamp = 2.0
        output_tensor = self.post_processor.postprocess(
            self.c, self.q_matrix, self.v_vector, lower_clamp, upper_clamp
        )

        # Check output is a tensor
        assert torch.is_tensor(output_tensor)

        # Check size is valid
        assert output_tensor.size() == self.c.size()

        # Check if pp time is valid
        error_message = "post_processing time must be greater than 0"
        self.assertGreater(self.post_processor.pp_time, 0, error_message)

    def test_postprocess_custom_num_iter_pp(self):
        """Test postprocess with custom num_iter_pp"""
        num_iter_pp = 10
        print("c", self.c)
        print("v_vector", self.v_vector)
        print("q_matrix", self.q_matrix)
        output_tensor = self.post_processor.postprocess(
            self.c, self.q_matrix, self.v_vector, num_iter_pp=num_iter_pp
        )
        print("output_tensor", output_tensor)

        self.assertTrue(torch.is_tensor(output_tensor))

        print("output_tensor.size()", output_tensor.size())
        print("self.c.size()", self.c.size())

        self.assertEqual(output_tensor.size(), self.c.size())

    def test_step_size_custom_step_size(self):
        """Test postprocess with custom step_size"""
        step_size = 0.05
        print("c2", self.c)
        print("v_vector2", self.v_vector)
        print("q_matrix2", self.q_matrix)
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
