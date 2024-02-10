import unittest
import numpy as np
import pandas as pd
from unittest import TestCase
from ccvm_simulators.ccvmplotlib.problem_metadata import ProblemType
from ccvm_simulators.ccvmplotlib.problem_metadata import BoxQPMetadata


class TestBoxQPMetadata(TestCase):
    """Test case for BoxQP Metadata class implementation."""

    def setUp(self) -> None:
        self.valid_problem_type = ProblemType.BoxQP
        self.invalid_problem_type_str = "CCVM"

        self.valid_metadata_filepath = (
            "ccvm_simulators/ccvmplotlib/tests/metadata/valid_metadata.json"
        )
        self.invalid_zero_perf_metadata_filepath = "ccvm_simulators/ccvmplotlib/tests/metadata/invalid_zero_performance_metadata.json"
        self.invalid_incorrect_field_metadata_filepath = "ccvm_simulators/ccvmplotlib/tests/metadata/invalid_incorrect_field_metadata.json"

        def valid_machine_func(matching_df: pd.DataFrame) -> float:
            return np.mean(matching_df["solve_time"].values, dtype=float)

        def invalid_machine_func() -> float:
            return

        self.valid_machine_func: callable = valid_machine_func
        self.invalid_machine_func: callable = invalid_machine_func

    def test_BoxQP_metadata_valid(self):
        """Test BoxQP Metadata class object when valid inputs are given."""
        boxqp_metadata = BoxQPMetadata(self.valid_problem_type)

        self.assertEqual(boxqp_metadata.problem, self.valid_problem_type)
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
            )

    def test_ingest_result_data_method_valid(self):
        """Test BoxQP Metadata class "ingest_result_data" method when valid
        inputs are given.
        """
        boxqp_metadata = BoxQPMetadata(self.valid_problem_type)
        boxqp_metadata.ingest_metadata(self.valid_metadata_filepath)

        self.assertGreater(boxqp_metadata._BoxQPMetadata__batch_size, 0)
        self.assertGreater(len(boxqp_metadata._BoxQPMetadata__problem_size_list), 0)

    def test_ingest_result_data_method_invalid(self):
        """Test BoxQP Metadata class "ingest_result_data" method when invalid
        inputs are given.
        """
        boxqp_metadata = BoxQPMetadata(self.valid_problem_type)
        with self.assertRaises(KeyError):
            boxqp_metadata.ingest_metadata(
                self.invalid_incorrect_field_metadata_filepath
            )

    def test_generate_TTS_plot_data_valid(self):
        """Test BoxQP Metadata class "generate_TTS_plot_data" method when valid
        inputs are given.
        """
        boxqp_metadata = BoxQPMetadata(self.valid_problem_type)
        boxqp_metadata.ingest_metadata(self.valid_metadata_filepath)
        TTS_plot_data = boxqp_metadata.generate_TTS_plot_data(
            machine_time_func=self.valid_machine_func
        )

        self.assertIsInstance(TTS_plot_data, pd.DataFrame)
        self.assertGreater(TTS_plot_data.size, 0)

    def test_generate_TTS_plot_data_invalid(self):
        """Test BoxQP Metadata class "generate_TTS_plot_data" method when
        invalid inputs are given.
        """
        boxqp_metadata = BoxQPMetadata(self.valid_problem_type)
        boxqp_metadata.ingest_metadata(self.valid_metadata_filepath)
        with self.assertRaises(TypeError):
            boxqp_metadata.generate_TTS_plot_data(
                machine_time_func=self.invalid_machine_func
            )

    def test_generate_success_prob_plot_data_valid(self):
        """Test BoxQP Metadata class "generate_success_prob_plot_data" method
        when valid inputs are given.

        # TODO: Implementation
        """
        pass

    def test_generate_success_prob_plot_data_invalid(self):
        """Test BoxQP Metadata class "generate_success_prob_plot_data" method
        when invalid inputs are given.

        # TODO: Implementation
        """
        pass


if __name__ == "__main__":
    unittest.main()
