import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import cm

from ccvm_simulators.ccvmplotlib.problem_metadata import ProblemMetadataFactory


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
        fig: matplotlib.figure.Figure = None,
        ax: matplotlib.axes.Axes = None,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plot a problem-specific Time-To-Solution metadata solved by a CCVM
        solver.

        Args:
            metadata_filepath (str): A file path to metadata.
            problem (str): A problem type.
            TTS_type (str): A Time-To-Solution type. It is either a CPU time or an
            optic device time.
            fig (matplotlib.figure.Figure, optional): A pre-generated pyplot figure. Defaults to None.
            ax (matplotlib.axes.Axes, optional): A pre-generated pyplot axis. Defaults to None.

        Raises:
            ValueError: Raises a ValueError when the plotting data is not valid.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Returns a figure and axis that has
                the TTS plot with minimal styling.
        """
        problem_metadata = ProblemMetadataFactory.create_problem_metadata(
            problem, TTS_type
        )
        problem_metadata.ingest_metadata(metadata_filepath)
        plotting_df = problem_metadata.generate_plot_data()

        x_data = plotting_df.index

        if not ax or not fig:
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
        fig: matplotlib.figure.Figure = None,
        ax: matplotlib.axes.Axes = None,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plot a problem-specific success probability result data solved by a
        CCVM solver.

        Args:
            metadata_filepath (str): A file path to solution data.
            problem (str): A problem type.
            TTS_type (str): A Time-To-Solution type. It is either a CPU time or an
            optic device time
            fig (matplotlib.figure.Figure, optional): A pre-generated pyplot figure. Defaults to None.
            ax (matplotlib.axes.Axes, optional): A pre-generated pyplot axis. Defaults to None.

        Raises:
            ValueError: Raises a ValueError when the plotting data is invalid.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Returns a figure and axis that has
                the success probability plot with minimal styling.
        """
        problem_metadata = ProblemMetadataFactory.create_problem_metadata(
            problem, TTS_type
        )
        problem_metadata.ingest_metadata(metadata_filepath)
        plotting_df = problem_metadata.generate_plot_data()
        x_data = plotting_df.index.tolist()

        if not ax or not fig:
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
    def set_default_figsize(fig: matplotlib.figure.Figure) -> None:
        """A method to set the figure size with default width and height values.

        Args:
            fig (matplotlib.figure.Figure): A pyplot figure to be resized.
        """
        fig.set_figwidth(8.0)
        fig.set_figheight(7.0)

    @staticmethod
    def set_default_xlabel(ax: matplotlib.axes.Axes, xlabel: str) -> None:
        """A method to set the x-label text. Also, it sets font and font size
        with default values.

        Args:
            ax (matplotlib.axes.Axes): A pyplot axis to be set.
            xlabel (str): x-label text.
        """
        ax.set_xlabel(xlabel=xlabel, fontdict={"family": "serif", "size": 36})

    @staticmethod
    def set_default_ylabel(ax: matplotlib.axes.Axes, ylabel: str) -> None:
        """A method to set the y-label text. Also, it sets font and font size
        with default values.

        Args:
            ax (matplotlib.axes.Axes): A pyplot axis to be set.
            ylabel (str): y-label text.
        """
        ax.set_ylabel(ylabel=ylabel, fontdict={"family": "serif", "size": 36})

    @staticmethod
    def set_default_ticks(ax: matplotlib.axes.Axes) -> None:
        """A method to set the x&y ticks with default font size.

        Args:
            ax (matplotlib.axes.Axes): A pyplot axis to be set.
        """
        ax.tick_params(axis="x", labelsize=32)
        ax.tick_params(axis="y", labelsize=32)

    @staticmethod
    def set_default_legend(ax: matplotlib.axes.Axes) -> None:
        """A method to set the legen with default configuration.

        Args:
            ax (matplotlib.axes.Axes): A pyplot axis to be set.
        """
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

    @staticmethod
    def set_default_grid(ax: matplotlib.axes.Axes) -> None:
        """A method to set the grid with default configuration.

        Args:
            ax (matplotlib.axes.Axes): A pyplot axis to be set.
        """
        ax.grid(
            visible=True,
            which="major",
            axis="both",
            color="#666666",
            linestyle="--",
        )

    @staticmethod
    def apply_default_tts_styling(
        fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes
    ) -> None:
        """A method to apply the default styling to a TTS plot.

        Args:
            fig (matplotlib.figure.Figure): A pyplot figure to be set.
            ax (matplotlib.axes.Axes): A pyplot axis to be set.
        """
        # set figure size
        ccvmplotlib.set_default_figsize(fig)

        # set x & y labels
        ccvmplotlib.set_default_xlabel(ax, "Problem Size, $N$")
        ccvmplotlib.set_default_ylabel(ax, "TTS (seconds)")

        # set x & y ticks
        ccvmplotlib.set_default_ticks(ax)

        # set legend
        ccvmplotlib.set_default_legend(ax)

        # set grid
        ccvmplotlib.set_default_grid(ax)

        # call tight layout
        fig.tight_layout()

    @staticmethod
    def apply_default_succ_prob_styling(
        fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes
    ) -> None:
        """A method to apply the default styling to a success probability plot.

        Args:
            fig (matplotlib.figure.Figure): A pyplot figure to be set.
            ax (matplotlib.axes.Axes): A pyplot axis to be set.
        """
        # set figure size
        ccvmplotlib.set_default_figsize(fig)

        # set x & y labels
        ccvmplotlib.set_default_xlabel(ax, "Problem Size, $N$")
        ccvmplotlib.set_default_ylabel(ax, "Success Probability")

        # set x & y ticks
        ccvmplotlib.set_default_ticks(ax)

        # set legend
        ccvmplotlib.set_default_legend(ax)

        # set grid
        ccvmplotlib.set_default_grid(ax)

        # call tight layout
        fig.tight_layout()
