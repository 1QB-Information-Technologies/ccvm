from ccvm.post_processor.PostProcessor import MethodType
from ccvm.post_processor.PostProcessorAdam import PostProcessorAdam
from ccvm.post_processor.PostProcessorASGD import PostProcessorASGD
from ccvm.post_processor.PostProcessorBFGS import PostProcessorBFGS
from ccvm.post_processor.PostProcessorLBFGS import PostProcessorLBFGS
from ccvm.post_processor.PostProcessorTrustConstr import PostProcessorTrustConstr


class PostProcessorFactory:
    """The Factory Class"""

    @staticmethod
    def create_postprocessor(method):
        """Create the relevant post processor from given method.

        Args:
            method (MethodType): The type of method for post-processing.

        Raises:
            AssertionError: Invalid method type is provided.

        Returns:
            PostProcessor: A post processor object depending on the given method.
        """
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
        raise AssertionError(f"Method type is not valid. Provided: {method}")
