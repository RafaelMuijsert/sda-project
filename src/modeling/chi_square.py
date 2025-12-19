"""Predicting obesity through hereditary and lifestyle behaviors.

Copyright (C) 2025.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)


def contingency_table(
    var1: pd.Series,
    var2: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a contingency table.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: the contingency table.

    """
    ct = pd.crosstab(var1, var2)
    ct_percent = pd.DataFrame.round(ct.div(ct.sum(axis=1), axis=0) * 100, 2)

    return ct, ct_percent


def chi_square_test(
    var1: pd.Series,
    var2: pd.Series,
) -> tuple[float, float, int, np.ndarray]:
    """Perform a chi-square test.

    Returns:
        tuple[float, float, int, np.ndarray]: the chi square test results.

    """
    ct = pd.crosstab(var1, var2)
    chi2, p, dof, ex = chi2_contingency(ct, correction=False)

    return chi2, p, dof, ex


def main() -> None:
    """Perform chi-square tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        filename="results/chi_square_summary.log",
        filemode="w",
    )
    df = pd.read_csv("data/obesity_cleaned_final.csv", sep=";")

    # --- Family History vs. Obese Binary ---
    ct_fh, _ = contingency_table(
        df["family_history_with_overweight"],
        df["Obese_Binary"],
    )
    logger.info("--- Contingency Table: Family History vs. Obese Binary ---")
    logger.info(ct_fh.to_string())

    chi2, p, dof, _ = chi_square_test(
        df["family_history_with_overweight"],
        df["Obese_Binary"],
    )
    logger.info(
        "\n--- Chi-square test: family_history_with_overweight and Obese_Binary ---",
    )
    logger.info(
        "Chi2 Statistic: %.4f, p-value: %.4f, Degrees of Freedom: %d",
        chi2,
        p,
        dof,
    )

    # --- FAVC vs. Obese Binary ---
    ct_favc, _ = contingency_table(df["FAVC"], df["Obese_Binary"])
    logger.info("\n--- Contingency Table: FAVC vs. Obese Binary ---")
    logger.info(ct_favc.to_string())

    chi2, p, dof, _ = chi_square_test(df["FAVC"], df["Obese_Binary"])
    logger.info("\n--- Chi-square test: FAVC and Obese_Binary ---")
    logger.info(
        "Chi2 Statistic: %.4f, p-value: %.4f, Degrees of Freedom: %d",
        chi2,
        p,
        dof,
    )


if __name__ == "__main__":
    main()
