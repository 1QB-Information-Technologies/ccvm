from ccvm.ccvmplotlib.problem_metadata.problem_metadata import (
    ProblemType,
    TTSType,
    ProblemMetadata,
)
from ccvm.ccvmplotlib.utils.sampleTTSmetric import SampleTTSMetric

import numpy as np
import pandas as pd
import json_stream


class BoxQPMetadata(ProblemMetadata):
    """BoxQP Problem-specific Metadata class.

    The problem-specific metadata class inherited from Problem Metadata parent
    class for the BoxQP problem. This takes Box QP problem result data and
    generates plotting data.
    """

    def __init__(self, problem: ProblemType, TTS_type: TTSType) -> None:
        """BoxQP Metadata class object constructor.

        The constructor defines variables that are specific to the BoxQP problem
        and are used to generate plotting data.

        Args:
            problem (ProblemType): A problem type.
            TTS_type (TTSType): A Time-To-Solve type. It is either a CPU time or an
                optic device time.
        """
        super().__init__(problem, TTS_type)
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
            for key in data_stream[0]["solution_performance"]:
                self.__percent_gap_list.append(key)

        # populate pd.DataFrame
        with open(metadata_filepath, "r") as test_file:
            data_stream = json_stream.load(test_file)
            for data in data_stream:
                self.__df = pd.concat(
                    [self.__df, pd.DataFrame([self.__flatten_dict(data)])],
                    ignore_index=True,
                )

        self.__batch_size = self.__df["batch_size"][0]
        self.__problem_size_list = sorted(self.__df["problem_size"].unique().tolist())

    def generate_arrays_of_TTS(
        self, matching_df: pd.DataFrame, percent_gap: str, problem_size: int
    ) -> list:
        """A method to generate more data points that is used for plotting.

        With a given result data in pandas.DataFrame, it generates best known
        energy list, Time-To-Solve list, and success probability list based on
        the optimal type and problem size inputs.

        Args:
            matching_df (pd.DataFrame): Filtered Data Frame with the solver parameter 
                and the problem size.
            percent_gap (str): Percentage gap from the optimal solution.
            problem_size (int): BoxQP problem size (N).

        Raises:
            AssertionError: Raises an error if an unsupported TTS type is given

        Returns:
            list: Best known energy list, Time-To-Solve list, and success probability 
                list
        """
        row_count = matching_df.shape[0]
        best_known_energy_arr = np.ones(row_count) * 10

        all_results = []
        for row in range(row_count):
            percent_counter = 0
            success_prob = float(matching_df[percent_gap].values[row])
            array_to_add = []
            success_prob_checker = int(
                success_prob * matching_df["batch_size"].values[row]
            )
            pp_time = float(matching_df["pp_time"].values[row])
            iterations = matching_df["iterations"].values[row]
            if super().TTS_type == TTSType.wallclock:
                machine_time = float(matching_df["solve_time"].values[row])
            elif super().TTS_type == TTSType.physical:
                machine_time = float(problem_size) * 10e-12 * iterations + pp_time
            else:
                raise AssertionError(f'"{super().TTS_type}" is not supported.')

            for _ in range(self.__batch_size):
                if percent_counter < success_prob_checker:
                    array_to_add.append({"best_energy": 10, "solve_time": machine_time})
                else:
                    array_to_add.append({"best_energy": 15, "solve_time": machine_time})
                percent_counter += 1
            all_results.append(array_to_add)

        return best_known_energy_arr, all_results, matching_df[percent_gap].values

    def generate_plot_data(self) -> pd.DataFrame:
        """A method to generate data for TTS and success probability plotting.

        Using the "generate_arrays_of_TTS()" method, it generates new data in
        the pandas.DataFrame and the output gets used for TTS and success
        probability plottings.

        Returns:
            pd.DataFrame: A new processed dataframe for TTS and success probability 
                plotting.
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
                        tau_attribute="solve_time",
                        percentile=int(percentile),
                        seed=1,
                        num_bootstraps=1000,
                    )
                    (
                        best_known_energy_arr,
                        result_list,
                        success_prob_arr,
                    ) = self.generate_arrays_of_TTS(
                        matching_df, percent_gap, problem_size
                    )

                    success_prob_mean = np.mean(
                        np.array(
                            [float(success_prob) for success_prob in success_prob_arr]
                        )
                    )
                    if len(best_known_energy_arr) == 0:
                        mean_TTS = np.inf
                    else:
                        mean_TTS, _ = sampler.calc(result_list, best_known_energy_arr)
                    plotting_df.at[problem_size, (percent_gap, percentile)] = mean_TTS

                plotting_df.at[
                    problem_size, (percent_gap, "success_prob")
                ] = success_prob_mean

        return plotting_df
