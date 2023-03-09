import numpy
from scipy.stats import beta as beta_distribution
import sys
from typing import Union

from .metric import Metric


class SampleTTSMetric(Metric):
    """Time to solution (TTS) metric."""

    def __init__(
        self,
        tau_attribute: str,
        percentile: float = 50.0,
        confidence: float = 0.99,
        num_bootstraps: int = 100,
        failure_fill_in_value: float = sys.float_info.max,
        tolerance: float = 1e-5,
        seed: int = None,
        **kwargs
    ):
        """Time to solution (TTS) metric.

        Args:
            tau_attribute (str): Name of result key to use as tau = the length of each
            run, tau is then set to the mean over the results.
            percentile (float, optional): The percentile of the R99 distribution to
            find. Defaults to 50.0.
            confidence (float, optional): The required confidence such as 0.99 for 99%
            confidence. Defaults to 0.99.
            num_bootstraps (int, optional): Number of bootstrapped samples.
            Defaults to 100.
            failure_fill_in_value (float, optional): Fill in value to replace the mean
            and std with (respectively) if they are NaN or inf, or None to leave as is.
            Defaults to sys.float_info.max.
            tolerance (float, optional): Tolerance for comparisons with
            best_known_energy. Defaults to 1e-5.
            seed (int, optional): Random seed for the bootstrapping process.
            Defaults to None.

        Keyword Args:
            Additional arguments to pass to the parent __init__ method.

        Raises:
            ValueError: Raises error if the confidence value is out of range.
        """
        super(SampleTTSMetric, self).__init__(**kwargs)
        self._rng = numpy.random.RandomState(seed)

        self.name = "tts"
        self.tau_attribute = tau_attribute
        self.percentile = percentile
        self.confidence = confidence
        self.num_bootstraps = num_bootstraps
        self.failure_fill_in_value = failure_fill_in_value
        self.tolerance = tolerance

        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1")

    def calc(
        self, results: list[list], best_known_energies: list, **kwargs
    ) -> list[float]:
        """Calculate the mean and std of the sample TTS with the given
        percentile.

        Args:
            results (list[list]): A result for each problem.
            best_known_energies (list): A best known energy for the problem associated 
                with each result.

        Keyword Args:
            Captures additional named arguments passed to Metric.calc().

        Returns:
            list[float]: The estimate of the given percentile of the TTS distribution,
            the std of the given percentile of the TTS distribution.
        """
        # Calculate the success probabilities
        success_probabilities = self.calc_success_probabilities(
            results, best_known_energies
        )

        # If less than percentile problems were solved, do not attempt
        # to estimate the TTS
        frac_solved = (success_probabilities > 0).mean()
        if frac_solved < self.percentile / 100.0:
            mean_TTS = numpy.inf
            std_TTS = numpy.inf

        else:
            # Find the number of repeats
            num_repeats = self.num_solutions_per_result(results)

            # Calculate the R99 sample, with this percentile
            R99_distribution = self.calc_R99_distribution(
                success_probabilities, num_repeats
            )

            # Mean and variance of tau
            mean_tau = self.overall_mean(results, self.tau_attribute)
            var_tau = self.overall_variance(results, self.tau_attribute)

            # Mean and variance of R99
            mean_R99 = numpy.mean(R99_distribution)
            var_R99 = numpy.var(R99_distribution)

            # Mean and std of TTS
            mean_TTS = mean_R99 * mean_tau
            std_TTS = (
                (var_R99 * var_tau)
                + (mean_R99**2 * var_tau)
                + (mean_tau**2 * var_R99)
            ) ** 0.5

        if self.failure_fill_in_value is not None:
            mean_TTS = self.fill_in_value(mean_TTS, self.failure_fill_in_value)
            std_TTS = self.fill_in_value(std_TTS, self.failure_fill_in_value)

        return mean_TTS, std_TTS

    def calc_R99(self, success_probability: float) -> float:
        """Calculate the R99.

        Given a set of solutions and the best known solution, calculate the R99,
        defined as the number of independent runs that we need to call the
        solver to find a solution as good as the best_known_energy (or better)
        at least once with 99% confidence.

        Args:
            success_probability (float): The success probability for solutions given best_known_energy.

        Raises:
            ValueError: Raises an error if the confidence value is out of range.

        Returns:
            float: The calcuated R99 value.
        """
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1")

        # Trivial cases
        if success_probability == 0:
            R99 = numpy.inf
        elif success_probability == 1:
            R99 = 1.0
        else:
            # Calculate the R99
            R99 = numpy.log(1 - self.confidence) / numpy.log(1 - success_probability)
            # Cannot be smaller than 1
            if R99 < 1:
                R99 = 1.0

        # Final check
        assert (R99 is numpy.inf) or (R99 >= 1.0)

        return R99

    def calc_R99_distribution(
        self, success_probabilities: list[float], num_repeats: int
    ) -> numpy.ndarray:
        """Returns the percentile-th R99 distribution for a sample of problems.

        Note: problems should be of the same size.

        Args:
            success_probabilities (list[float]): A success probability for each problem.
            num_repeats (int): Number of repeats in each result.

        Returns:
            numpy.ndarray: An R99 distribution from bootstrapping.
        """
        # Find the Beta posterior distribution
        beta_posterior = []
        for success_probability in success_probabilities:
            # The parameters of the Beta posterior distribution The prior is
            # assumed to be a Beta distribution with alpha = 0.5 and beta = 0.5
            alpha = 0.5 + success_probability * num_repeats  # 0.5 + number of successes
            beta = (
                0.5 + (1 - success_probability) * num_repeats
            )  # 0.5 + number of failures
            beta_posterior.append((alpha, beta))

        # The percentile R99 empirical distribution, one percentile of R99 for
        # each bootstrapped sample
        R99_distribution = numpy.empty(self.num_bootstraps, dtype=float)
        # For each bootstrap sample TODO: check default for num_bootstraps: 100
        # might be too few, maybe 500 or 1000 (?)
        for i in range(self.num_bootstraps):
            # Construct a sample of R99's of the same length as success_probabilities
            R99_sampled = []

            # Choose the random indices into beta_posterior for the below loop
            # in advance
            random_indices = self._rng.randint(
                0, len(beta_posterior), len(success_probabilities)
            )
            # Choose the random numbers for the cdf_values in the below loop, in advance
            cdf_values = self._rng.uniform(0, 1, len(success_probabilities))
            for random_index, cdf_value in zip(random_indices, cdf_values):
                # Generate the beta random variable given the cdf_value
                sampled_probability = float(
                    beta_distribution.ppf(cdf_value, *beta_posterior[random_index])
                )

                # Calculate the R99 and add it to the list of R99s for this bootstrap
                R99_sampled.append(self.calc_R99(sampled_probability))

            # Calculate the percentile of the bootstrap sample and store it
            R99_distribution[i] = numpy.percentile(R99_sampled, self.percentile)

        # Return distribution of R99s
        return R99_distribution

    def calc_success_probabilities(
        self, results: list, best_known_energies: list
    ) -> numpy.ndarray:
        """Calculate the success probabilities for all problems.

        Args:
            results (list): Results for each problem.
            best_known_energies (list): A best known energy for the problem associated with each result

        Returns:
            numpy.ndarray: A success probability for each problem.
        """
        probabilities = numpy.empty(len(results), dtype=float)
        for i, (result, energy) in enumerate(zip(results, best_known_energies)):
            probabilities[i] = self.calc_success_probability(result, energy)
        return probabilities

    def calc_success_probability(
        self, solutions: Union[list, dict], best_known_energy: float
    ) -> float:
        """Calculate the success probability for a given problem.

        Args:
            solutions (Union[list, dict]): The solutions.
            best_known_energy (float): The lowest energy known.

        Returns:
            float: Success probability.
        """
        n_success = sum(
            1
            for solution in solutions
            if solution["best_energy"] < best_known_energy + self.tolerance
        )
        return n_success / float(len(solutions))
