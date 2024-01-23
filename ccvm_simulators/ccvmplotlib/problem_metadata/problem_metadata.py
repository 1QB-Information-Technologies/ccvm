from abc import ABC, abstractmethod
import pandas as pd
from enum import Enum


class ProblemType(Enum):
    """Problem type ENUM class."""

    BoxQP = "BoxQP"


class TTSType(Enum):
    """Time-To-Solution Type ENUM class.

    Either a CPU time (physical) or an optic device time (wallclock).
    """

    wallclock = "wallclock"
    physical = "physical"


class ProblemMetadata(ABC):
    """Abstract class for the problem metadata."""

    def __init__(self, problem: ProblemType) -> None:
        """Problem Metadata abstract class object constructor.

        The constructor defines and holds common variables for different
        problems solved with CCVM solvers.

        Args:
            problem (ProblemType): A problem type.
        """
        self.__problem = problem

    @property
    def problem(self) -> ProblemType:
        """Problem type getter method.

        Returns:
            ProblemType: Problem Type.
        """
        return self.__problem

    @abstractmethod
    def ingest_metadata(self) -> None:
        """Take a file path to metadata and convert them into a
        pandas.DataFrame."""

    @abstractmethod
    def generate_TTS_plot_data(self) -> pd.DataFrame:
        """Generate data for plotting.

        Plotting data will be varied by a different problem. Thus, for different
        problems, plotting data preparation code needs to be implemented in its
        inherited proble-specific metadata object.

        Returns:
            pd.DataFrame: A new data for plotting.
        """
        pass

    @abstractmethod
    def generate_success_prob_plot_data(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        pass
