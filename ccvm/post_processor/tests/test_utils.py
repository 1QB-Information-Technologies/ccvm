from unittest import TestCase
from ccvm.post_processor.utils import *
import torch


class BoxQPTest(TestCase):
    def setUp(self):
        # TODO: Implementation
        c = torch.Tensor([0.5, 0.5])
        self.box_qp = BoxQP(c)

    def test_forward_valid(self):
        """Test forward when given valid inputs"""
        # TODO: Implementation
        pass


class UtilsTest(TestCase):
    def test_func_post_valid(self):
        """Test func_post when given valid inputs"""
        # TODO: Implementation
        pass

    def test_func_post_jac_valid(self):
        """Test func_post_jac when given valid inputs"""
        # TODO: Implementation
        pass

    def test_func_post_hess_valid(self):
        """Test func_post_hess when given valid inputs"""
        # TODO: Implementation
        pass

    def test_func_post_LBFGS_valid(self):
        """Test func_post_LBFGS when given valid inputs"""
        # TODO: Implementation
        pass

    def test_func_post_torch_valid(self):
        """Test func_post_torch when given valid inputs"""
        # TODO: Implementation
        pass
