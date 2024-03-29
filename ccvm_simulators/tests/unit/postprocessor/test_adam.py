import logging
import torch
from unittest import TestCase
from ccvm_simulators.post_processor.adam import PostProcessorAdam


class TestPostProcessorAdam(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger()
        cls.post_processor = PostProcessorAdam()
        cls.N = 20
        cls.M = 100
        cls.c = torch.zeros(cls.M, cls.N)
        cls.q_matrix_asym = torch.randint(-50, 50, [cls.N, cls.N], dtype=torch.float)
        cls.v_vector = torch.randint(-50, 50, [cls.N], dtype=torch.float)
        cls.q_matrix = (
            torch.triu(cls.q_matrix_asym) + torch.triu(cls.q_matrix_asym, diagonal=1).T
        )

    def setUp(self):
        self.logger.info("Test %s Started" % (self._testMethodName))

    def tearDown(self):
        self.logger.info("Test %s Finished" % (self._testMethodName))

    def test_postprocess_valid(self):
        """Test postprocess when given valid inputs and verified the pp_time gets
        updated correctly
        """

        output_tensor = self.post_processor.postprocess(
            self.c, self.q_matrix, self.v_vector
        )

        # Check output is a tensor
        assert torch.is_tensor(output_tensor)

        # Check size is valid
        assert output_tensor.size() == self.c.size()

        # Check if pp time is valid
        error_message = "post_processing time must be greater than 0"
        self.assertGreater(self.post_processor.pp_time, 0, error_message)

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

    def test_postprocess_invalid_c_parameter(self):
        """Test postprocess when c value is not a tensor"""
        invalid_c = "dummy-c"

        with self.assertRaisesRegex(TypeError, "parameter c must be a tensor"):
            self.post_processor.postprocess(invalid_c, self.q_matrix, self.v_vector)

    def test_postprocess_invalid_qmat_parameter(self):
        """Test postprocess when qmat value is not a tensor"""
        invalid_qmat = "dummy-qmat"

        with self.assertRaisesRegex(TypeError, "parameter q_matrix must be a tensor"):
            self.post_processor.postprocess(self.c, invalid_qmat, self.v_vector)

    def test_postprocess_invalid_v_vector_parameter(self):
        """Test postprocess when v_vector value is not a tensor"""
        invalid_v_vector = "dummy-v_vector"

        with self.assertRaisesRegex(TypeError, "parameter v_vector must be a tensor"):
            self.post_processor.postprocess(self.c, self.q_matrix, invalid_v_vector)

    def test_postprocess_error_for_invalid_c_dimension(self):
        """Test postprocess when parameter dimensions are inconsistent.
        We expect to be given an MxN tensor for c, an NxN tensor for q_matrix, and
        a tensor of size N for the v_vector. If any of these are not the correct
        size, we expect an exception to be raised
        """

        # N is 20 and the incompatible_dimension is not equals to N (20)
        incompatible_dimension = 9
        c = torch.FloatTensor(self.M, incompatible_dimension)

        try:
            self.post_processor.postprocess(c, self.q_matrix, self.v_vector)
        except Exception:
            pass
        else:
            self.fail("Expected Exception not raised")

    def test_postprocess_error_for_invalid_v_vector_shape(self):
        """Test postprocess when parameter dimensions are inconsistent.
        We expect to be given an MxN tensor for c, an NxN tensor for q_matrix, and
        a tensor of size N for the v_vector. If any of these are not the correct
        size, we expect an exception to be raised
        """
        N = self.N
        v_vector = torch.FloatTensor(N, N)

        try:
            self.post_processor.postprocess(self.c, self.q_matrix, v_vector)
        except Exception:
            pass
        else:
            self.fail("Expected Exception not raised")
