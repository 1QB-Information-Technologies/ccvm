import unittest
from unittest.mock import Mock
from unittest import TestCase
from ccvm_simulators.ccvmplotlib.problem_metadata import (
    ProblemMetadataFactory,
    ProblemType,
    TTSType,
)
from ccvm_simulators.ccvmplotlib.problem_metadata import BoxQPMetadata


class TestProblemMetadataFactory(TestCase):
    """Test Problem Metadata Factory class implementation."""

    def setUp(self) -> None:
        self.valid_problem_type = ProblemType.BoxQP
        self.invalid_problem_type = "CCVM"
        self.TTS_type = TTSType.wallclock
        self.boxqp_metadata = BoxQPMetadata(self.valid_problem_type, self.TTS_type)

    def test_problem_metadata_factory_valid(self):
        """Test Problem Metadata Factory class when valid inputs are given."""
        problem_metadata = ProblemMetadataFactory.create_problem_metadata(
            self.valid_problem_type, self.TTS_type
        )

        self.assertEqual(problem_metadata.__class__.__name__, "BoxQPMetadata")
        self.assertEqual(problem_metadata.problem, self.valid_problem_type)
        self.assertEqual(problem_metadata.TTS_type, self.TTS_type)
        self.assertEqual(
            problem_metadata._BoxQPMetadata__problem_size_list,
            self.boxqp_metadata._BoxQPMetadata__problem_size_list,
        )
        self.assertEqual(
            problem_metadata._BoxQPMetadata__percent_gap_list,
            self.boxqp_metadata._BoxQPMetadata__percent_gap_list,
        )
        self.assertEqual(
            problem_metadata._BoxQPMetadata__percentile_list,
            self.boxqp_metadata._BoxQPMetadata__percentile_list,
        )
        self.assertEqual(
            problem_metadata._BoxQPMetadata__batch_size,
            self.boxqp_metadata._BoxQPMetadata__batch_size,
        )
        self.assertTrue(problem_metadata._BoxQPMetadata__df.empty)

    def test_problem_metadata_factory_invalid_input(self):
        """Test Problem Metadata Factory class when invalid inputs are given."""
        with self.assertRaises(ValueError):
            ProblemMetadataFactory.create_problem_metadata(
                self.invalid_problem_type, self.TTS_type
            )


if __name__ == "__main__":
    unittest.main()
