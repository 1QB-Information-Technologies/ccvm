from .post_processor import MethodType
from .adam import PostProcessorAdam
from .asgd import PostProcessorASGD
from .bfgs import PostProcessorBFGS
from .lbfgs import PostProcessorLBFGS
from .trust_constr import PostProcessorTrustConstr


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
