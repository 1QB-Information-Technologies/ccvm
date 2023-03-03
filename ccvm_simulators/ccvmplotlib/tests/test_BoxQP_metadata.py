import unittest
from unittest import TestCase
from ccvm_simulators.ccvmplotlib.problem_metadata import ProblemType, TTSType
from ccvm_simulators.ccvmplotlib.problem_metadata import BoxQPMetadata


class TestBoxQPMetadata(TestCase):
    """Test case for BoxQP Metadata class implementation."""

    def setUp(self) -> None:
        self.valid_problem_type = ProblemType.BoxQP
        self.invalid_problem_type_str = "CCVM"
        self.valid_TTS_type = TTSType.physical
        self.invalid_TTS_type_str = "wall_clock"

    def test_BoxQP_metadata_valid(self):
        """Test BoxQP Metadata class object when valid inputs are given."""
        boxqp_metadata = BoxQPMetadata(self.valid_problem_type, self.valid_TTS_type)

        self.assertEqual(boxqp_metadata.problem, self.valid_problem_type)
        self.assertEqual(boxqp_metadata.TTS_type, self.valid_TTS_type)
        self.assertEqual(boxqp_metadata._BoxQPMetadata__problem_size_list, [])
        self.assertEqual(boxqp_metadata._BoxQPMetadata__percent_gap_list, [])
        self.assertEqual(
            boxqp_metadata._BoxQPMetadata__percentile_list,
            ["25", "50", "75", "success_prob"],
        )
        self.assertEqual(boxqp_metadata._BoxQPMetadata__batch_size, 0)
        self.assertTrue(boxqp_metadata._BoxQPMetadata__df.empty)

    def test_BoxQP_metadata_invalid_problem_type(self):
        """Test BoxQP Metadata class object when an invalid problem type is
        given.
        """
        with self.assertRaises(ValueError):
            BoxQPMetadata(
                ProblemType(self.invalid_problem_type_str),
                self.valid_TTS_type,
            )

    def test_BoxQP_metadata_invalid_TTS_type(self):
        """Test BoxQP Metadata class object when an invalid TTS type is
        given.


        """
        with self.assertRaises(ValueError):
            BoxQPMetadata(self.valid_problem_type, TTSType(self.invalid_TTS_type_str))

    def test_ingest_result_data_method_valid(self):
        """Test BoxQP Metadata class "ingest_result_data" method when valid
        inputs are given.

        # TODO: Implementation
        """
        pass

    def test_ingest_result_data_method_invalid(self):
        """Test BoxQP Metadata class "ingest_result_data" method when invalid
        inputs are given.

        # TODO: Implementation
        """
        pass

    def test_generate_arrays_of_TTS_method_valid(self):
        """Test BoxQP Metadata class "generate_arrays_of_TTS" method when valid
        inputs are given.

        # TODO: Implementation
        """
        pass

    def test_generate_arrays_of_TTS_method_invalid(self):
        """Test BoxQP Metadata class "generate_arrays_of_TTS" method when
        invalid inputs are given.

        # TODO: Implementation
        """
        pass

    def test_generate_plot_data_method(self):
        """Test BoxQP Metadata class "generate_plot_data" method.

        # TODO: Implementation
        """
        pass


if __name__ == "__main__":
    unittest.main()
