import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Any
from matplotlib import cm

from ccvm.ccvmplotlib.problem_metadata import ProblemMetadataFactory


TTS_UPPER_LIMIT = 1e20  # Approximate age of the universe in sec.
PERC_GAP_LABEL_MAP = {
    "optimal": r"0.1% gap",
    "one_percent": r"1% gap",
    "two_percent": r"2% gap",
    "three_percent": r"3% gap",
    "four_percent": r"4% gap",
    "five_percent": r"5% gap",
    "ten_percent": r"10% gap",
}


class ccvmplotlib:
    """A generic plotting library for a problem solved by a CCVM solver."""

    @staticmethod
    def plot_TTS(
        metadata_filepath: str,
        problem: str,
        TTS_type: str,
        ax: matplotlib.axes.Axes | np.ndarray = None,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | np.ndarray]:
        """Plot a problem-specific Time-To-Solution metadata solved by a CCVM
        solver.

        Args:
            metadata_filepath (str): A file path to metadata.
            problem (str): A problem type.
            TTS_type (str): A Time-To-Solution type. It is either a CPU time or an
            optic device time.
            ax (matplotlib.axes.Axes | np.ndarray, optional): _description_. Defaults to None.

        Raises:
            ValueError: Raises a ValueError when the plotting data is not valid.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | np.ndarray]: _description_
        """
        problem_metadata = ProblemMetadataFactory.create_problem_metadata(
            problem, TTS_type
        )
        problem_metadata.ingest_metadata(metadata_filepath)
        plotting_df = problem_metadata.generate_plot_data()

        x_data = plotting_df.index

        if not ax:
            fig, ax = plt.subplots()

        color_iter = cm.rainbow(np.linspace(0, 1, len(plotting_df.columns.levels[0])))
        for lvl0_column_name, color in zip(plotting_df.columns.levels[0], color_iter):
            # Plotting IQR
            ax.fill_between(
                x_data,
                list(plotting_df[lvl0_column_name, "25"]),
                list(plotting_df[lvl0_column_name, "75"]),
                color=color,
                alpha=0.2,
            )
            # Plotting Median
            ax.plot(
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

        ax.plot(
            [],
            [],
            linestyle="-",
            marker="s",
            label="(median)",
            color="black",
            linewidth=4.0,
        )
        ax.fill_between([], [], alpha=0.2, label="(IQR)")

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

        ax.set_ylim(lower_lim, upper_lim)  # limit on y values
        ax.set_yscale("log")  # log scale

        # Make sure x-axis only has integer values
        ax.set_xticks(x_data)

        return (fig, ax)

    @staticmethod
    def plot_success_prob(
        metadata_filepath: str,
        problem: str,
        TTS_type: str,
        ax: matplotlib.axes.Axes | np.ndarray = None,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | np.ndarray]:
        """Plot a problem-specific success probability result data solved by a
        CCVM solver.

        Args:
            metadata_filepath (str): A file path to solution data.
            problem (str): A problem type.
            TTS_type (str): A Time-To-Solution type. It is either a CPU time or an
            optic device time
            ax (matplotlib.axes.Axes | np.ndarray, optional): _description_. Defaults to None.

        Raises:
            ValueError: Raises a ValueError when the plotting data is invalid.

        Returns:
            matplotlib.figure.Figure: A figure object of the plotted success
            probability result data.
        """
        problem_metadata = ProblemMetadataFactory.create_problem_metadata(
            problem, TTS_type
        )
        problem_metadata.ingest_metadata(metadata_filepath)
        plotting_df = problem_metadata.generate_plot_data()
        x_data = plotting_df.index.tolist()

        if not ax:
            fig, ax = plt.subplots()

        color_iter = cm.rainbow(np.linspace(0, 1, len(plotting_df.columns.levels[0])))
        max_succ_prob = -np.inf
        for lvl0_column_name, color in zip(plotting_df.columns.levels[0], color_iter):
            max_succ_prob = max(
                max_succ_prob, np.max(plotting_df[lvl0_column_name, "success_prob"])
            )
            ax.plot(
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

        ax.set_yscale("log")

        # Make sure x-axis only has integer values
        ax.set_xticks(x_data)

        return (fig, ax)

    @staticmethod
    def set_figsize(fig: matplotlib.figure.Figure, fig_width: float, fig_height: float) -> None:
        fig.set_figwidth(fig_width)
        fig.set_figheight(fig_height)

    @staticmethod
    def set_xlabel(ax: matplotlib.axes.Axes, xlabel: str, font_size: float = 36) -> None:
        ax.set_xlabel(xlabel=xlabel, fontsize=font_size)

    @staticmethod
    def set_ylabel(ax: matplotlib.axes.Axes, ylabel: str, font_size: float = 36) -> None:
        ax.set_ylabel(ylabel=ylabel, fontsize=font_size)

    @staticmethod
    def apply_default_styling(fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes) -> None:
        # set figure size
        fig.set_figwidth(7.7)
        fig.set_figheight(7.0)

        # set x & y labels
        ax.set_xlabel("Problem Size, $N$", fontdict={'family':'serif', 'size':36})
        ax.set_ylabel("TTS (seconds)", fontdict={'family':'serif', 'size':36})

        # set x & y ticks
        ax.tick_params(axis='x', labelsize=32)
        ax.tick_params(axis='y', labelsize=32)

        # set legend
        handles, labels = plt.gca().get_legend_handles_labels()
        label_list = list(PERC_GAP_LABEL_MAP.values())
        label_list.extend(["(median)", "(IQR)"])
        legend_orders = []
        for label in label_list:
            try:
                legend_orders.append(labels.index(label))
            except Exception:
                pass
        ax.legend(
            [handles[idx] for idx in legend_orders],
            [labels[idx] for idx in legend_orders],
            loc="best",
            ncol=2,
        )

        # set grid
        ax.grid(
            visible=True,
            which="major",
            axis="both",
            color="#666666",
            linestyle="--",
        )

        # call tight layout
        fig.tight_layout()