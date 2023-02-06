import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Any
from matplotlib import cm

from ccvm.ccvmplotlib.problem_metadata import ProblemMetadataFactory


TTS_UPPER_LIMIT = 1e20  # Approximate age of the universe in sec.
PERC_GAP_LABEL_MAP = {
    "optimal": r"0.1\% gap",
    "one_percent": r"1\% gap",
    "two_percent": r"2\% gap",
    "three_percent": r"3\% gap",
    "four_percent": r"4\% gap",
    "five_percent": r"5\% gap",
    "ten_percent": r"10\% gap",
}


class ccvmplotlib:
    """A generic plotting library for a problem solved by a CCVM solver."""

    @staticmethod
    def plot_TTS(
        solution_filepath: str,
        problem: str,
        TTS_type: str,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Plot a problem-specific Time-To-Solution result data solved by a CCVM
        solver.

        Args:
            solution_filepath (str): A file path to solution data.
            problem (str): A problem type.
            TTS_type (str): A Time-To-Solution type. It is either a CPU time or an
            optic device time.

        Keyword Args:
            ylim (Tuple[float, float]): Set the y-limits of a plot. (e.g. ylim =
            (lower_lim, upper_lim))

            show_plot (bool): When set to True, the plot will pop up in a new window
            when it is ready.

        Raises:
            ValueError: Raises a ValueError when the plotting data is not valid.

        Returns:
            matplotlib.figure.Figure: A figure object of the plotted Time-To-Solution
            result data.
        """
        problem_metadata = ProblemMetadataFactory.create_problem_metadata(
            problem, TTS_type
        )
        problem_metadata.ingest_solution_data(solution_filepath)
        plotting_df = problem_metadata.generate_plot_data()

        x_data = plotting_df.index

        figure = plt.figure(figsize=(7.7, 7.0))
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        fonts = {"xlabel": 36, "ylabel": 36, "legend": 26, "xticks": 32, "yticks": 32}

        color_iter = cm.rainbow(np.linspace(0, 1, len(plotting_df.columns.levels[0])))
        for lvl0_column_name, color in zip(plotting_df.columns.levels[0], color_iter):
            # Plotting IQR
            plt.fill_between(
                x_data,
                list(plotting_df[lvl0_column_name, "25"]),
                list(plotting_df[lvl0_column_name, "75"]),
                color=color,
                alpha=0.2,
            )
            # Plotting Median
            plt.plot(
                x_data,
                plotting_df[lvl0_column_name, "50"],
                linestyle="-",
                marker="s",
                label=PERC_GAP_LABEL_MAP[lvl0_column_name]
                if lvl0_column_name in PERC_GAP_LABEL_MAP
                else lvl0_column_name,
                color=color,
                linewidth=4.0,
            )

        plt.xlabel("Problem Size $N$", fontsize=fonts["xlabel"])
        plt.ylabel("TTS (seconds)", fontsize=fonts["ylabel"])
        plt.plot(
            [],
            [],
            linestyle="-",
            marker="s",
            label="(median)",
            color="black",
            linewidth=4.0,
        )
        plt.fill_between([], [], alpha=0.2, label="(IQR)")

        if "ylim" in kwargs:
            lower_lim = kwargs["ylim"][0]
            upper_lim = kwargs["ylim"][1]
        else:
            # Get max & min median TTS values
            min_median = np.inf
            max_median = -np.inf
            for lvl0_column in plotting_df.columns.levels[0]:
                min_median = min(min_median, np.min(plotting_df[lvl0_column, "50"]))
                max_median = max(max_median, np.max(plotting_df[lvl0_column, "50"]))

            if min_median >= TTS_UPPER_LIMIT:
                raise ValueError(
                    f"TTS values are too large to plot. Please check the result data. Minimum TTS median value: {min_median}"
                )
            else:
                upper_lim = 10 ** (
                    math.ceil(np.log10(min(min_median * (1e6), max_median))) + 1
                )
                lower_lim = 10 ** (math.floor(np.log10(min_median)) - 1)

        plt.ylim(lower_lim, upper_lim)  # limit on y values
        plt.yscale("log")  # log scale
        plt.grid(
            visible=True,
            which="major",
            axis="both",
            color="#666666",
            linestyle="--",
        )  # grid lines on the graph

        handles, labels = plt.gca().get_legend_handles_labels()
        label_list = list(PERC_GAP_LABEL_MAP.values())
        label_list.extend(["(median)", "(IQR)"])
        legend_orders = []
        for label in label_list:
            try:
                legend_orders.append(labels.index(label))
            except Exception:
                pass
        plt.legend(
            [handles[idx] for idx in legend_orders],
            [labels[idx] for idx in legend_orders],
            loc="best",
            ncol=2,
        )

        plt.xticks(fontsize=fonts["xticks"])
        plt.yticks(fontsize=fonts["yticks"])
        plt.tight_layout()

        if "show_plot" in kwargs and kwargs["show_plot"]:
            plt.show()
        return figure

    @staticmethod
    def plot_success_prob(
        solution_filepath: str,
        problem: str,
        TTS_type: str,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Plot a problem-specific success probability result data solved by a
        CCVM solver.

        Args:
            solution_filepath (str): A file path to solution data.
            problem (str): A problem type.
            TTS_type (str): A Time-To-Solution type. It is either a CPU time or an
            optic device time

        Keyword Args:
            ylim (Tuple[float, float]): Set the y-limits of a plot. (e.g. ylim =
            (lower_lim, upper_lim))
            show_plot (bool): When set to True, the plot will pop up in a new window
            when it is ready.

        Raises:
            ValueError: Raises a ValueError when the plotting data is invalid.

        Returns:
            matplotlib.figure.Figure: A figure object of the plotted success
            probability result data.
        """
        problem_metadata = ProblemMetadataFactory.create_problem_metadata(
            problem, TTS_type
        )
        problem_metadata.ingest_solution_data(solution_filepath)
        plotting_df = problem_metadata.generate_plot_data()
        x_data = plotting_df.index.tolist()

        figure = plt.figure(figsize=(7.7, 7.0))
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        fonts = {"xlabel": 36, "ylabel": 36, "legend": 26, "xticks": 32, "yticks": 32}

        color_iter = cm.rainbow(np.linspace(0, 1, len(plotting_df.columns.levels[0])))
        max_succ_prob = -np.inf
        for lvl0_column_name, color in zip(plotting_df.columns.levels[0], color_iter):
            max_succ_prob = max(
                max_succ_prob, np.max(plotting_df[lvl0_column_name, "success_prob"])
            )
            plt.plot(
                x_data,
                plotting_df[lvl0_column_name, "success_prob"],
                linestyle="-",
                marker="s",
                label=PERC_GAP_LABEL_MAP[lvl0_column_name]
                if lvl0_column_name in PERC_GAP_LABEL_MAP
                else lvl0_column_name,
                color=color,
            )
        if max_succ_prob == 0.0:
            raise ValueError(
                "Success Probability values are all 0.0. Please check the result data."
            )

        plt.grid(
            visible=True,
            which="major",
            axis="both",
            color="#666666",
            linestyle="--",
        )

        handles, labels = plt.gca().get_legend_handles_labels()
        label_list = list(PERC_GAP_LABEL_MAP.values())
        legend_orders = []
        for label in label_list:
            try:
                legend_orders.append(labels.index(label))
            except Exception:
                pass
        plt.legend(
            [handles[idx] for idx in legend_orders],
            [labels[idx] for idx in legend_orders],
            loc="best",
            ncol=2,
        )

        if "ylim" in kwargs:
            upper_lim = kwargs["ylim"][0]
            lower_lim = kwargs["ylim"][1]
            plt.ylim(lower_lim, upper_lim)  # limit on y values

        plt.yscale("log")
        plt.xlabel("Problem Size, $N$", fontsize=fonts["xlabel"])
        plt.ylabel("Success Probability", fontsize=fonts["ylabel"])

        plt.xticks(fontsize=fonts["xticks"])
        plt.yticks(fontsize=fonts["yticks"])
        plt.tight_layout()

        if "show_plot" in kwargs and kwargs["show_plot"]:
            plt.show()

        return figure
