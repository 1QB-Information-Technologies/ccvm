from ccvm_simulators.ccvmplotlib.problem_metadata.problem_metadata import (
    ProblemMetadata,
    TTSType,
    ProblemType,
)
from ccvm_simulators.ccvmplotlib.problem_metadata import BoxQPMetadata


class ProblemMetadataFactory:
    """A Factory class to create a problem specific metadata class."""

    @staticmethod
    def create_problem_metadata(problem: str, TTS_type: str) -> ProblemMetadata:
        """A method to create a problem-specific Metadata class.

        This is a factory method that identifies the type of a problem and
        produces an appropriate Problem Metadata object.

        Args:
            problem (str): A problem type.
            TTS_type (str): A Time-To-Solution type. It is either a CPU time or an
                optic device time.

        Raises:
            AssertionError: Raises an error if an unsupported problem is given.

        Returns:
            ProblemMetadata: A problem-specific Metadata object.
        """
        try:
            if ProblemType(problem) == ProblemType.BoxQP:
                return BoxQPMetadata(ProblemType(problem), TTSType(TTS_type))
            else:
                raise AssertionError(f'"{problem}" problem type is not supported.')
        except AssertionError as e:
            print(e)
