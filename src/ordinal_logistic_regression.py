"""Ordinary logistic regression.

Copyright (C) 2025.
"""

import pandas as pd
from statsmodels.miscmodels.ordinal_model import (
    OrderedModel,
    OrderedResults,
)

DATASET_PATH = "data/obesity_cleaned_final.csv"


def read_data(filename: str) -> pd.DataFrame:
    """Read the dataset.

    Returns:
        pd.DataFrame: the dataset as a pandas dataframe.

    """
    return pd.read_csv(filename, sep=";")


def prepare_for_ordinal_regression(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the dataset for ordinal logistic regression.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: the original target and the predictors

    """
    # Ordinal target
    y: pd.DataFrame = df["Obesity_Level"]

    # Predictors
    x: pd.DataFrame = df[
        [
            "Gender",
            "Age",
            "CALC",
            "MTRANS_Automobile",
            "MTRANS_Bike",
            "MTRANS_Motorbike",
            "MTRANS_Public_Transportation",
            # MTRANS_Walking is DROPPED (baseline) because it made the sum of MTRANS=1
        ]
    ]

    return x, y


def run_ordinal_logistic_regression(
    x: pd.DataFrame,
    y: pd.DataFrame,
) -> OrderedResults:
    """Run ordinal logistich regression on the given target and predictors.

    Returns:
        OrderedResultsWrapper: the results of the ordinary logistic regression

    """
    model = OrderedModel(y, x, distr="logit")

    return model.fit(method="bfgs")


def main() -> None:
    """Read and prepare the dataset and perform ordinal logistic regression."""
    df = read_data(DATASET_PATH)

    x, y = prepare_for_ordinal_regression(df)

    result = run_ordinal_logistic_regression(x, y)

    print(result.summary())  # noqa: T201


if __name__ == "__main__":
    main()
