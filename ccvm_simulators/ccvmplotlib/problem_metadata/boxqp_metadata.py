from ccvm_simulators.ccvmplotlib.problem_metadata.problem_metadata import (
    ProblemType,
    ProblemMetadata,
)
from ccvm_simulators.ccvmplotlib.utils.sampleTTSmetric import SampleTTSMetric

import numpy as np
import pandas as pd
import json_stream


class BoxQPMetadata(ProblemMetadata):
    """BoxQP Problem-specific Metadata class.

    The problem-specific metadata class inherited from Problem Metadata parent
    class for the BoxQP problem. This takes Box QP problem result data and
    generates plotting data.
    """

    def __init__(self, problem: ProblemType) -> None:
        """BoxQP Metadata class object constructor.

        The constructor defines variables that are specific to the BoxQP problem
        and are used to generate plotting data.

        Args:
            problem (ProblemType): A problem type.
        """
        super().__init__(problem)
        self.__problem_size_list: list[int] = []
        self.__percent_gap_list: list[str] = []
        self.__percentile_list: list[str] = ["25", "50", "75", "success_prob"]
        self.__batch_size: int = 0
        self.__df: pd.DataFrame = pd.DataFrame()

    def __flatten_dict(self, result: dict) -> dict:
        """Flatten a nested dictionary.

        Args:
            result (dict): Result in the nested dictionary.

        Returns:
            dict: Flattened dictionary.
        """
        flattened_dict = {}
        for key_depth0, val_depth0 in result.items():
            if isinstance(val_depth0, dict) or isinstance(
                val_depth0, json_stream.base.TransientStreamingJSONObject
            ):
                for key_depth1, val_depth1 in val_depth0.items():
                    flattened_dict[key_depth1] = val_depth1
            else:
                flattened_dict[key_depth0] = val_depth0

        return flattened_dict

    def ingest_metadata(self, metadata_filepath: str) -> None:
        """A method to ingest raw metadata.

        Take a file path to metadata and convert them into a pandas.DataFrame.

        Args:
            metadata_filepath (str): A file path to metadata.
        """
        # populate percent gap list
        with open(metadata_filepath, "r") as test_file:
            data_stream = json_stream.load(test_file)
            for key in data_stream["result_metadata"][0]["solution_performance"]:
                self.__percent_gap_list.append(key)

        # populate pd.DataFrame
        with open(metadata_filepath, "r") as test_file:
            data_stream = json_stream.load(test_file)
            for data in data_stream["result_metadata"]:
                self.__df = pd.concat(
                    [self.__df, pd.DataFrame([self.__flatten_dict(data)])],
                    ignore_index=True,
                )

        self.__batch_size = self.__df["batch_size"][0]
        self.__problem_size_list = sorted(self.__df["problem_size"].unique().tolist())

    def generate_plot_data(
        self,
        metric_func: callable,
    ) -> pd.DataFrame:
        """Calculate the time to solution vs problem size for a particular gap and
        quantile.

        Args:
            metric_func (callable): A callback function that is used when calculating
            the metrics either to determine the `machine_time` or the `energy_max`,
            which are used when computing the TTS or ETS, respectively.
        Returns:
            (pd.Series): The time to solution for each problem size.
        """

        plotting_df = pd.DataFrame(
            index=pd.Index(self.__problem_size_list, name="Problem Size (N)"),
            columns=pd.MultiIndex.from_product(
                [self.__percent_gap_list, self.__percentile_list],
                names=["Optimality Type", "Percentile"],
            ),
        )

        for percent_gap in self.__percent_gap_list:
            for problem_size in self.__problem_size_list:
                matching_df = self.__df.loc[self.__df["problem_size"] == problem_size]
                for percentile in self.__percentile_list[:-1]:
                    sampler = SampleTTSMetric(
                        tau_attribute="time",
                        percentile=int(percentile),
                        seed=1,
                        num_bootstraps=100,
                    )

                    metric_value = metric_func(
                        matching_df=matching_df, problem_size=problem_size
                    )

                    success_prob = matching_df[percent_gap].values
                    frac_solved = (success_prob > 0).mean()
                    if frac_solved < (int(percentile) / 100):
                        R99 = np.inf
                    else:
                        R99_distribution = sampler.calc_R99_distribution(
                            success_probabilities=success_prob,
                            num_repeats=self.__batch_size,
                        )
                        R99 = np.mean(R99_distribution)

                    mean_metric = metric_value * R99
                    plotting_df.at[problem_size, (percent_gap, percentile)] = (
                        mean_metric
                    )

        return plotting_df

    def generate_success_prob_plot_data(self) -> pd.DataFrame:
        """Calculate the success probability vs problem size for a particular gap and
        quantile.

        Returns:
            pd.DataFrame: The success probability for each problem size.
        """
        plotting_df = pd.DataFrame(
            index=pd.Index(self.__problem_size_list, name="Problem Size (N)"),
            columns=pd.MultiIndex.from_product(
                [self.__percent_gap_list, self.__percentile_list],
                names=["Optimality Type", "Percentile"],
            ),
        )

        for percent_gap in self.__percent_gap_list:
            for problem_size in self.__problem_size_list:
                matching_df = self.__df.loc[self.__df["problem_size"] == problem_size]

                success_prob_list = matching_df[percent_gap].values
                mean_success_prob = np.mean(
                    np.array(
                        [float(success_prob) for success_prob in success_prob_list]
                    )
                )

                plotting_df.at[problem_size, (percent_gap, "success_prob")] = (
                    mean_success_prob
                )

        return plotting_df
