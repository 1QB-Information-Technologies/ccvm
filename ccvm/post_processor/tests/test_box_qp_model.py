from unittest import TestCase
import logging
from ..box_qp_model import BoxQPModel
from ..post_processor import MethodType
import torch
import numpy as np


class TestBoxQPModel(TestCase):
    @classmethod
    def setUpClass(self):
        self.logger = logging.getLogger()

        self.N = 20
        self.M = 100
        self.c = torch.FloatTensor(self.M, self.N)
        self.q_matrix = torch.FloatTensor(self.N, self.N)
        self.v_vector = torch.FloatTensor(self.N)

    def setUp(self):
        self.logger.info("Test %s Started" % (self._testMethodName))

    def tearDown(self):
        self.logger.info("Test %s Finished" % (self._testMethodName))

    def test_invalid_boxqp(self):
        """Test correct error is are raised when invalid method type is passed"""

        method_type = "test"
        self.boxqp = BoxQPModel(self.c, method_type)
        with self.assertRaisesRegex(
            ValueError,
            f"""Invalid method type provided for generating the model.
                Provided: {method_type}. Valid methods are {MethodType.Adam},
                {MethodType.ASGD} and {MethodType.LBFGS}.""",
        ):
            self.boxqp.forward(self.q_matrix, self.v_vector)

    def test_valid_boxqp_adam(self):
        "Test a valid tensor is returned when correct parameters are passed to forward function for adam post-processor"

        method_type = MethodType.Adam
        self.boxqp = BoxQPModel(self.c, method_type)
        output = self.boxqp.forward(self.q_matrix, self.v_vector)
        assert torch.is_tensor(output)

    def test_valid_boxqp_asgd(self):

        "Test a valid tensor is returned when correct parameters are passed to forward function for asgd post-processor"

        method_type = MethodType.ASGD
        self.boxqp = BoxQPModel(self.c, method_type)
        output = self.boxqp.forward(self.q_matrix, self.v_vector)
        assert torch.is_tensor(output)
