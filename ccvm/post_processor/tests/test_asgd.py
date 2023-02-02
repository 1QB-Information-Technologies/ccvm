from unittest import TestCase
import logging
from ..asgd import PostProcessorASGD
import torch
import numpy as np


class TestPostProcessorASGD(TestCase):
    @classmethod
    def setUpClass(self):
        self.logger = logging.getLogger()
        self.post_processor = PostProcessorASGD()

        self.N = 20
        self.M = 100
        self.c = torch.FloatTensor(self.M, self.N)
        self.q_mat = torch.FloatTensor(self.N, self.N)
        self.c_vector = torch.FloatTensor(self.N)

    def setUp(self):
        self.logger.info("Test %s Started" % (self._testMethodName))

    def tearDown(self):
        self.logger.info("Test %s Finished" % (self._testMethodName))

    def test_postprocess_valid(self):
        """Test postprocess when given valid inputs and verified the pp_time gets
        updated correctly
        """
        output_tensor = self.post_processor.postprocess(
            self.c, self.q_mat, self.c_vector
        )
        # check output is a tensor
        assert torch.is_tensor(output_tensor)
        # check size of valid
        assert output_tensor.size() == self.c.size()
        # check if pp time is valid
        error_message = "post_processing time must be greater than 0"
        self.assertGreater(self.post_processor.pp_time, 0, error_message)

    # TODO: Not sure if this is an applicable test case
    def test_postprocess_invalid_c_parameter(self):

        """Test postprocess when given valid inputs and verified the pp_time gets
        updated correctly
        """
        invalid_c = "dummy-c"
        with self.assertRaisesRegex(TypeError, "parameter c must be a tensor"):
            self.post_processor.postprocess(invalid_c, self.q_mat, self.c_vector)

    def test_postprocess_invalid_qmat_parameter(self):

        """Test postprocess when qmat value is not a tensor"""
        invalid_qmat = "dummy-qmat"

        with self.assertRaisesRegex(TypeError, "parameter q_mat must be a tensor"):
            self.post_processor.postprocess(self.c, invalid_qmat, self.c_vector)

    def test_postprocess_invalid_c_vector_parameter(self):

        """Test postprocess when c_vector value is not a tensor"""
        invalid_c_vector = "dummy-c_vector"

        with self.assertRaisesRegex(TypeError, "parameter c_vector must be a tensor"):
            self.post_processor.postprocess(self.c, self.q_mat, invalid_c_vector)

    def test_postprocess_error_for_invalid_c_dimension(self):

        """Test postprocess when parameter dimensions are inconsistent.
        We expect to be given an MxN tensor for c, an NxN tensor for q_mat, and
        a tensor of size N for the c_vector. If any of these are not the correct
        size, we expect an exception to be raised
        """

        # N is 20 and the incompatible_dimension is not equals to N (20)
        incompatible_dimension = 9
        c = torch.FloatTensor(self.M, incompatible_dimension)
        try:
            self.post_processor.postprocess(c, self.q_mat, self.c_vector)
        except Exception:
            pass
        else:
            self.fail("Expected Exception not raised")

    def test_postprocess_error_for_invalid_c_vector_shape(self):

        """Test postprocess when parameter dimensions are inconsistent.
        We expect to be given an MxN tensor for c, an NxN tensor for q_mat, and
        a tensor of size N for the c_vector. If any of these are not the correct
        size, we expect an exception to be raised
        """
        N = self.N
        c_vector = torch.FloatTensor(N, N)
        try:
            self.post_processor.postprocess(self.c, self.q_mat, c_vector)
        except Exception:
            pass
        else:
            self.fail("Expected Exception not raised")
