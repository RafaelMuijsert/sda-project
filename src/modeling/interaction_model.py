"""Analyze interaction effects between hereditary + lifestyle factors on obesity levels.

Copyright (C) 2025.
"""

import logging

import pandas as pd
from statsmodels.miscmodels.ordinal_model import (
    OrderedModel,
    OrderedResults,
)

logger = logging.getLogger(__name__)


DATASET_PATH = "data/obesity_cleaned_final.csv"


def read_data(filename: str) -> pd.DataFrame:
    """Read the dataset.

    Returns:
        pd.DataFrame: the dataset as a pandas dataframe.

    """
    return pd.read_csv(filename, sep=";")


def prepare_for_ordinal_regression_with_interactions(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the dataset for ordinal logistic regression with interaction terms.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: the original target and the predictors

    """
    y: pd.DataFrame = df["Obesity_Level"]

    # Base predictors
    predictors = [
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

    x = df[predictors].copy()

    # Define key lifestyle factors for interaction
    # These were chosen based on their significance in the previous lifestyle model
    lifestyle_interaction_vars = ["FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "FAF"]

    # Create interaction terms
    for col in lifestyle_interaction_vars:
        x[f"family_history_x_{col}"] = x["family_history_with_overweight"] * x[col]

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
    """Read + prepare dataset, perform ordinal logistic regression + interactions."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        filename="results/interaction_model_summary.log",
        filemode="w",
    )
    df = read_data(DATASET_PATH)

    x, y = prepare_for_ordinal_regression_with_interactions(df)

    result = run_ordinal_logistic_regression(x, y)

    logger.info("--- Interaction Model Summary ---")
    logger.info(result.summary().as_text())


if __name__ == "__main__":
    main()
