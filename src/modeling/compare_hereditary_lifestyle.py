"""Compare hereditary and lifestyle factors using ordinal logistic regression.

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        filename="results/hereditary_vs_lifestyle_summary.log",
        filemode="w",
    )
    df = read_data(DATASET_PATH)

    x, y = prepare_for_ordinal_regression(df)

    hereditary_predictors = x[["family_history_with_overweight"]]
    hereditary_result = run_ordinal_logistic_regression(hereditary_predictors, y)
    logger.info("--- Hereditary Model Summary ---")
    logger.info(hereditary_result.summary().as_text())

    lifestyle_predictors = x.drop(columns=["family_history_with_overweight"])
    lifestyle_result = run_ordinal_logistic_regression(lifestyle_predictors, y)
    logger.info("\n--- Lifestyle Model Summary ---")
    logger.info(lifestyle_result.summary().as_text())

    combined_result = run_ordinal_logistic_regression(x, y)
    logger.info("\n--- Combined Model Summary ---")
    logger.info(combined_result.summary().as_text())

    logger.info("\n--- Model Comparison (AIC/BIC) ---")
    logger.info(
        "Hereditary Model: AIC = %.2f, BIC = %.2f",
        hereditary_result.aic,
        hereditary_result.bic,
    )
    logger.info(
        "Lifestyle Model:  AIC = %.2f, BIC = %.2f",
        lifestyle_result.aic,
        lifestyle_result.bic,
    )
    logger.info(
        "Combined Model:   AIC = %.2f, BIC = %.2f",
        combined_result.aic,
        combined_result.bic,
    )
    logger.info("\nLower AIC/BIC values indicate a better model fit.")


if __name__ == "__main__":
    main()
