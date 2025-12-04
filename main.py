"""Predicting obesity through hereditary and lifestyle behaviors.

Copyright (C) 2025.
"""

import kagglehub
import pandas as pd

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

    contingency_table: pd.DataFrame = pd.crosstab(
        df.family_history_with_overweight,
        df.NObeyesdad,
    )

    print("First 5 records:", df.head())
    print(df.columns.values)
    print(contingency_table)


def main() -> None:
    """Load dataset and perform experiments."""
    load_dataset()


if __name__ == "__main__":
    main()
