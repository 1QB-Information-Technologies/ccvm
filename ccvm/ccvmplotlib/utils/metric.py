import numpy

from .mixins import StrDictMixIn
from . import utilities


class Metric(StrDictMixIn, object):
    """Parent Metric class: inherit all other metrics from this class."""

    def __init__(self, goal="minimize"):
        """Initialize the metric."""
        # The goal for this metric: 'minimize' or 'maximize'
        self.goal = goal

    def calc(self, results, best_known_energies, **kwargs):
        """Placeholder: calculate the metric value."""
        pass

    @staticmethod
    def overall_mean(results, key):
        """Calculate the overall average of the quantity that corresponds to key."""
        iterator = (el[key] for result in results for el in result)
        return utilities.imean(iterator)

    @staticmethod
    def overall_variance(results, key):
        """Calculate the variance of the quantity that corresponds to key."""
        iterator = (el[key] for result in results for el in result)
        return utilities.ivariance(iterator)

    @staticmethod
    def num_solutions_per_result(results: list[list]) -> int:
        """Return the number of solutions per result.

        All results must have the same number of solutions.

        Args:
            results (list[list]): List of result list.

        Raises:
            ValueError: Raises an error if number of result for each problem is 
                different.

        Returns:
            int: Return the only number of results for every result in the given list.
        """
        if len(results) == 0:
            return 0

        num_solutions = None
        for result in results:
            if not num_solutions:
                num_solutions = len(result)
            else:
                if num_solutions != len(result):
                    raise ValueError("Number of solutions not the same for all results")

        return num_solutions

    @staticmethod
    def fill_in_value(value: float, failure_fill_in_value: float) -> float:
        """If value is nan/inf, replace it with failure_fill_in_value.

        Args:
            value (float): The value to be checked.
            failure_fill_in_value (float): The replacement value on failure.

        Returns:
            float: Filled in value.
        """
        if numpy.isnan(value) or numpy.isinf(value):
            return failure_fill_in_value

        return value
