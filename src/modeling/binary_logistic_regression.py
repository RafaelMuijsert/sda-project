"""Perform binary logistic regression analysis.

Copyright (C) 2025.
"""

import pandas as pd
import statsmodels.api as sm

DATASET_PATH = "data/obesity_cleaned_final.csv"


def read_data(filename: str) -> pd.DataFrame:
    """Read the dataset.

    Returns:
        pd.DataFrame: the dataset as a pandas dataframe.

    """
    return pd.read_csv(filename, sep=";")


def prepare_for_binary_regression(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the dataset for binary logistic regression.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: the original target and the predictors

    """
    # Binary target
    y: pd.DataFrame = df["Obese_Binary"]

    # Predictors
    x: pd.DataFrame = df[
        [
            "Gender",
            "Age",
            "Height",
            "Weight",
            "family_history_with_overweight",
            "FAVC",
            "FCVC",
            "NCP",
            "CAEC",
            "SMOKE",
            "CH2O",
            "SCC",
            "FAF",
            "TUE",
            "CALC",
            "MTRANS_Automobile",
            "MTRANS_Bike",
            "MTRANS_Motorbike",
            "MTRANS_Public_Transportation",
        ]
    ]

    return x, y


def run_binary_logistic_regression(
    x: pd.DataFrame,
    y: pd.DataFrame,
) -> sm.Logit:
    """Run binary logistic regression on the given target and predictors.

    Returns:
        sm.Logit: the results of the binary logistic regression

    """
    x = sm.add_constant(x)
    model = sm.Logit(y, x)

    return model.fit()


def main() -> None:
    """Read and prepare the dataset and perform binary logistic regression."""
    df = read_data(DATASET_PATH)

    x, y = prepare_for_binary_regression(df)

    result = run_binary_logistic_regression(x, y)

    print(result.summary())  # noqa: T201


if __name__ == "__main__":
    main()
