"""Predicting obesity through hereditary and lifestyle behaviors.

Copyright (C) 2025.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CONTINGENCY_PLOT_PATH = "results/contingency.png"
HEATMAP_PLOT_PATH = "results/heatmap.png"


def plot_contingency(df: pd.DataFrame) -> None:
    """Create a contingency table and plot different visualizations."""
    # Create a temporary column for display of family history as 'Yes'/'No'
    df["family_history_display"] = df["family_history_with_overweight"].map(
        {0: "No", 1: "Yes"},
    )
    contingency_table: pd.DataFrame = pd.crosstab(
        df.family_history_display,
        df.Obesity_Category,
    )

    contingency_table.plot(kind="bar", figsize=(10, 6))

    plt.xlabel("Family History of Overweight")
    plt.ylabel("Count")
    plt.legend(
        title="Obesity Category",
    )
    plt.plot()

    plt.savefig(CONTINGENCY_PLOT_PATH)


def plot_heatmap(df: pd.DataFrame) -> None:
    """Plot a heatmap."""
    # Create a temporary column for display of family history as 'Yes'/'No'
    df["family_history_display"] = df["family_history_with_overweight"].map(
        {0: "No", 1: "Yes"},
    )
    # Heatmap
    df = pd.crosstab(
        df.family_history_display,
        df.Obesity_Category,
    )

    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Obesity Category")
    plt.ylabel("Family history")
    plt.tight_layout()
    plt.savefig(HEATMAP_PLOT_PATH)


def main() -> None:
    """Load dataset and perform experiments."""
    df = pd.read_csv("data/obesity_cleaned_final.csv", sep=";")
    plot_heatmap(df.copy())
    plot_contingency(df.copy())


if __name__ == "__main__":
    main()
