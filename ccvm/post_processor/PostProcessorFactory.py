from ccvm.post_processor.PostProcessorAdam import PostProcessorAdam
from ccvm.post_processor.PostProcessorASGD import PostProcessorASGD
from ccvm.post_processor.PostProcessorBFGS import PostProcessorBFGS
from ccvm.post_processor.PostProcessorLBFGS import PostProcessorLBFGS
from ccvm.post_processor.PostProcessorTrustConstr import PostProcessorTrustConstr
from enum import Enum


class MethodType(str, Enum):
    BFGS = "bfgs"
    TrustConst = "trust-constr"
    LBFGS = "lbfgs"
    Adam = "adam"
    ASGD = "asgd"

class PostProcessorFactory:
    """The Factory Class"""

    @staticmethod
    def create_postprocessor(method):
        if method.lower() == MethodType.BFGS.value:
            return PostProcessorBFGS()
        elif method.lower() == MethodType.TrustConst.value:
            return PostProcessorTrustConstr()
        elif method.lower() == MethodType.LBFGS.value:
            return PostProcessorLBFGS()
        elif method.lower() == MethodType.Adam.value:
            return PostProcessorAdam()
        elif method.lower() == MethodType.ASGD.value:
            return PostProcessorASGD()
        raise AssertionError("Method type is not valid.")
