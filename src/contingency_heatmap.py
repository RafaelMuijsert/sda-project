"""Predicting obesity through hereditary and lifestyle behaviors.

Copyright (C) 2025.
"""

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CONTINGENCY_PLOT_PATH = "results/contingency.png"
HEATMAP_PLOT_PATH = "results/heatmap.png"


def get_obesity_level_str(level: int) -> str:
    """Get the obesity level from an integer 0-6 as a readable string.

    Raises:
        ValueError: if the provided level is not a valid obesity level.

    Returns:
        str: the obesity level as a readable string.

    """
    levels: dict[int, str] = {
        0: "Underweight",
        1: "Normal Weight",
        2: "Overweight level 1",
        3: "Overweight level 2",
        4: "Obesity type 1",
        5: "Obesity type 2",
        6: "Obesity type 3",
    }
    if level not in levels:
        error = "Invalid obesity level"
        raise ValueError(error)

    return levels[level]


def clean_dataset(df: pd.DataFrame) -> None:
    """Clean up the dataset for contingency visualization."""
    # Remap so that the labels are clear
    df["family_history_with_overweight"] = df.family_history_with_overweight.map(
        {0: "No", 1: "Yes"},
    )


def plot_contingency(df: pd.DataFrame) -> None:
    """Create a contingency table and plot different visualizations."""
    contingency_table: pd.DataFrame = pd.crosstab(
        df.family_history_with_overweight,
        df.NObeyesdad,
    )

    contingency_table.plot(kind="bar", figsize=(10, 6))
    codes = contingency_table.columns.tolist()

    plt.xlabel("Family History of Overweight")
    plt.ylabel("Count")
    plt.legend(
        [get_obesity_level_str(code) for code in codes],
        title="Obesity Level",
    )
    plt.plot()

    plt.savefig(CONTINGENCY_PLOT_PATH)


def plot_heatmap(df: pd.DataFrame) -> None:
    """Plot a heatmap."""
    # Heatmap
    df = pd.crosstab(
        df.family_history_with_overweight,
        df.NObeyesdad,
    )
    df.columns = [get_obesity_level_str(int(x)) for x in df.columns]

    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Obesity level")
    plt.ylabel("Family history")
    plt.tight_layout()
    plt.savefig(HEATMAP_PLOT_PATH)


def load_dataset() -> None:
    """Load the dataset from kaggle."""
    # Set the path to the file you'd like to load
    file_path = (
        "estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.csv"
    )

    # Load the latest version
    df: pd.DataFrame = kagglehub.dataset_load(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType, reportUnknownMemberType]
        kagglehub.KaggleDatasetAdapter.PANDAS,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        "ikjotsingh221/obesity-risk-prediction-cleaned",
        file_path,
    )

    clean_dataset(df)
    plot_heatmap(df)
    plot_contingency(df)


def main() -> None:
    """Load dataset and perform experiments."""
    load_dataset()


if __name__ == "__main__":
    main()
