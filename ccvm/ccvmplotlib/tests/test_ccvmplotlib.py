import matplotlib
import matplotlib.pyplot as plt
import unittest
from unittest import TestCase
from ccvm.ccvmplotlib.ccvmplotlib import ccvmplotlib


# # TODO: Speed up tests
# https://github.com/1QB-Information-Technologies/ccvm/issues/34
# class TestCCVMPlotLib(TestCase):
#     """Test ccvmplotlib class."""

#     def setUp(self) -> None:
#         self.valid_metadata_filepath = "ccvm/ccvmplotlib/tests/metadata/valid_metadata.json"
#         # Below is an invalid result data since all 'solution_performance' values
#         # are set to 0.0.
#         self.invalid_metadata_filepath = "ccvm/ccvmplotlib/tests/metadata/invalid_metadata.json"
#         self.valid_problem_type_str = "BoxQP"
#         self.invalid_problem_type_str = "CCVM"
#         self.valid_TTS_type_str = "wallclock"
#         self.invalid_TTS_type_str = "physicals"

#     def test_plot_TTS_valid(self):
#         """Test ccvmplotlib class plot_TTS method when valid inputs are given."""
#         plot_fig, plot_ax = ccvmplotlib.plot_TTS(
#             self.valid_metadata_filepath,
#             self.valid_problem_type_str,
#             self.valid_TTS_type_str,
#         )
#         self.assertIsInstance(plot_fig, matplotlib.figure.Figure)
#         self.assertIsInstance(plot_ax, matplotlib.axes.Axes)

#     def test_plot_TTS_invalid_metadata(self):
#         """Test ccvmplotlib class plot_TTS method when an invalid result is
#         given.
#         """
#         with self.assertRaises(ValueError):
#             ccvmplotlib.plot_TTS(
#                 self.invalid_metadata_filepath,
#                 self.valid_problem_type_str,
#                 self.valid_TTS_type_str,
#             )

#     def test_plot_TTS_invalid_problem_type(self):
#         """Test ccvmplotlib class plot_TTS method when an invalid problem type
#         is given.
#         """
#         with self.assertRaises(ValueError):
#             ccvmplotlib.plot_TTS(
#                 self.valid_metadata_filepath,
#                 self.invalid_problem_type_str,
#                 self.valid_TTS_type_str,
#             )

#     def test_plot_TTS_invalid_TTS_type(self):
#         """Test ccvmplotlib class plot_TTS method when an invalid TTS type is
#         given.
#         """
#         with self.assertRaises(ValueError):
#             ccvmplotlib.plot_TTS(
#                 self.valid_metadata_filepath,
#                 self.valid_problem_type_str,
#                 self.invalid_TTS_type_str,
#             )

#     def test_plot_success_prob_valid(self):
#         """Test ccvmplotlib class plot_success_prob method when valid inputs are
#         given."""
#         plot_fig, plot_ax = ccvmplotlib.plot_success_prob(
#             self.valid_metadata_filepath,
#             self.valid_problem_type_str,
#             self.valid_TTS_type_str,
#         )
#         self.assertIsInstance(plot_fig, matplotlib.figure.Figure)
#         self.assertIsInstance(plot_ax, matplotlib.axes.Axes)

#     def test_plot_success_prob_invalid_metadata(self):
#         """Test ccvmplotlib class plot_success_prob method when an invalid
#         result is given."""
#         with self.assertRaises(ValueError):
#             ccvmplotlib.plot_success_prob(
#                 self.invalid_metadata_filepath,
#                 self.valid_problem_type_str,
#                 self.valid_TTS_type_str,
#             )

#     def test_plot_success_prob_invalid_problem_type(self):
#         """Test ccvmplotlib class plot_success_prob method an invalid problem
#         type is given.
#         """
#         with self.assertRaises(ValueError):
#             ccvmplotlib.plot_success_prob(
#                 self.valid_metadata_filepath,
#                 self.invalid_problem_type_str,
#                 self.valid_TTS_type_str,
#             )

#     def test_plot_success_prob_invalid_TTS_type(self):
#         """Test ccvmplotlib class plot_success_prob method an invalid TTS type
#         is given.
#         """
#         with self.assertRaises(ValueError):
#             ccvmplotlib.plot_success_prob(
#                 self.valid_metadata_filepath,
#                 self.valid_problem_type_str,
#                 self.invalid_TTS_type_str,
#             )


# if __name__ == "__main__":
#     unittest.main()
