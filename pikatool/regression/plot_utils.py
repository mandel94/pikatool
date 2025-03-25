import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Any


class ConfidenceIntervalPlotter:
    """
    A class to generate confidence interval plots for different categories.

    This class takes a DataFrame with lower and upper confidence interval bounds
    and visualizes them using matplotlib.

    :Example:

        >>> conf_intervals_df = pd.DataFrame({'Lower Bound': [1.2, 2.5, 3.0, 4.1],
        ...                                   'Upper Bound': [2.0, 3.5, 4.1, 5.2]},
        ...                                  index=['A', 'B', 'C', 'D'])
        >>> plotter = ConfidenceIntervalPlotter(conf_intervals_df)
        >>> plotter.plot()
    """

    def __init__(
        self,
        conf_intervals_df: pd.DataFrame,
        theme: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the ConfidenceIntervalPlotter.

        :param conf_intervals_df: A DataFrame with two columns (lower and upper confidence interval bounds).
        :param theme: A dictionary containing styling options for the plot. Defaults to None.
        :raises ValueError: If the DataFrame format is incorrect.
        """
        self._validate_input(conf_intervals_df)
        self.categories: np.ndarray = conf_intervals_df.index.to_numpy()
        self.lower_bounds: np.ndarray = conf_intervals_df.iloc[:, 0].to_numpy()
        self.upper_bounds: np.ndarray = conf_intervals_df.iloc[:, 1].to_numpy()

        # Default theme settings
        self.theme = {
            "figsize": (12, 6),
            "marker_color": "#008150",
            "zero_line_color": "#E3001D",
            "grid_color": "#bcbcbc",
            "marker_size": 8,
            "capsize": 6,
            "capthick": 2,
            "marker_edge_width": 1.5,
            "title_fontsize": 16,
            "title_fontweight": "bold",
            "title_color": "#333333",
            "label_fontsize": 14,
            "label_fontweight": "regular",
            "label_color": "#4a4a4a",
            "tick_fontsize": 12,
            "tick_fontweight": "light",
            "tick_color": "#4a4a4a",
            "grid_alpha": 0.3,
            "spine_visibility": {"top": False, "right": False},
            "legend_fontsize": 12,
            "legend_location": "upper left",
            "tight_layout": True,
        }

        # Update theme if custom settings are provided
        if theme:
            self.theme.update(theme)

    def _validate_input(self, df: pd.DataFrame) -> None:
        """
        Validates that the input DataFrame has the correct format.

        :param df: The DataFrame to validate.
        :raises ValueError: If the DataFrame is not properly formatted.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame.")
        if df.shape[1] != 2:
            raise ValueError(
                "DataFrame must have exactly two columns (lower and upper bounds)."
            )
        if not isinstance(df.index, pd.Index):
            raise ValueError("DataFrame index must contain category names.")

    def _compute_intervals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the means and errors of confidence intervals.

        :return: A tuple containing means and error margins.
        """
        means: np.ndarray = (self.lower_bounds + self.upper_bounds) / 2
        errors: np.ndarray = means - self.lower_bounds  # Compute error margins
        return means, errors

    def _sort_data(
        self, means: np.ndarray, errors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sorts means, categories, and errors in descending order.

        :param means: Array of mean values.
        :param errors: Array of error margins.
        :return: Tuple of sorted means, categories, and errors.
        """
        sorted_indices = np.argsort(-means)
        return (
            means[sorted_indices],
            self.categories[sorted_indices],
            errors[sorted_indices],
        )

    def plot(
        self,
        title: str = "Confidence Intervals for Categories",
        xlabel: str = "Values",
        ylabel: str = "Categories",
    ) -> None:
        """
        Generates the confidence interval plot.

        :param title: Title of the plot.
        :param xlabel: Label for the x-axis.
        :param ylabel: Label for the y-axis.
        """
        means, errors = self._compute_intervals()
        sorted_means, sorted_categories, sorted_errors = self._sort_data(
            means, errors
        )

        # Set figure size
        plt.figure(figsize=self.theme["figsize"])

        # Plot error bars
        plt.errorbar(
            sorted_means,
            sorted_categories,
            xerr=sorted_errors,
            fmt="o",
            capsize=self.theme["capsize"],
            capthick=self.theme["capthick"],
            label="Confidence Interval",
            color=self.theme["marker_color"],
            markersize=self.theme["marker_size"],
            linestyle="None",
            markeredgewidth=self.theme["marker_edge_width"],
        )

        # Add a vertical zero line
        plt.axvline(
            x=0,
            color=self.theme["zero_line_color"],
            linestyle="--",
            linewidth=2,
            label="Zero Line",
        )

        # Customize plot appearance
        plt.xlabel(
            xlabel,
            fontsize=self.theme["label_fontsize"],
            fontweight=self.theme["label_fontweight"],
            color=self.theme["label_color"],
            labelpad=15,
        )
        plt.ylabel(
            ylabel,
            fontsize=self.theme["label_fontsize"],
            fontweight=self.theme["label_fontweight"],
            color=self.theme["label_color"],
            labelpad=15,
        )
        plt.title(
            title,
            fontsize=self.theme["title_fontsize"],
            fontweight=self.theme["title_fontweight"],
            color=self.theme["title_color"],
            pad=20,
        )

        # Customize ticks
        plt.xticks(
            fontsize=self.theme["tick_fontsize"],
            fontweight=self.theme["tick_fontweight"],
            color=self.theme["tick_color"],
        )
        plt.yticks(
            fontsize=self.theme["tick_fontsize"],
            fontweight=self.theme["tick_fontweight"],
            color=self.theme["tick_color"],
        )

        # Remove specified spines

        for spine, visible in self.theme["spine_visibility"].__dict__.items():
            plt.gca().spines[spine].set_visible(visible)

        # Add horizontal grid lines
        plt.grid(
            True,
            linestyle="-",
            alpha=self.theme["grid_alpha"],
            color=self.theme["grid_color"],
        )

        # Reverse y-axis to put the first category on top
        plt.gca().invert_yaxis()

        # Add a legend
        plt.legend(
            frameon=False,
            fontsize=self.theme["legend_fontsize"],
            loc=self.theme["legend_location"],
        )

        # Ensure proper layout
        if self.theme["tight_layout"]:
            plt.tight_layout()

        # Show the plot
        plt.show()
