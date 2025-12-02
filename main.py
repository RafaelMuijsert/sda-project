"""Predicting obesity through hereditary and lifestyle behaviors.

Copyright (C) 2025.
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter


def load_dataset() -> None:
    """Load the dataset from kaggle."""
    # Set the path to the file you'd like to load
    file_path = (
        "estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.csv"
    )

    # Load the latest version
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "ikjotsingh221/obesity-risk-prediction-cleaned",
        file_path,
    )

    print("First 5 records:", df.head())


def main() -> None:
    """Load dataset and perform experiments."""
    load_dataset()


if __name__ == "__main__":
    main()
