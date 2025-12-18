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
    """Create a contingency table."""
    ct = pd.crosstab(var1, var2)
    ct_percent = pd.DataFrame.round(ct.div(ct.sum(axis=1), axis=0) * 100, 2)

    return ct, ct_percent


def chi_square_test(
    var1: pd.Series,
    var2: pd.Series,
) -> tuple[float, float, int, np.ndarray]:
    """Perform a chi-square test."""
    ct = pd.crosstab(var1, var2)
    chi2, p, dof, ex = chi2_contingency(ct, correction=False)

    return chi2, p, dof, ex


def main() -> None:
    """Perform chi-square tests."""
    df = pd.read_csv("data/obesity_cleaned_final.csv", sep=";")

    chi2, p, dof, _ = chi_square_test(
        df["family_history_with_overweight"],
        df["Obese_Binary"],
    )
    logger.info(
        "Chi-square test between family_history_with_overweight and Obese_Binary:",
    )
    logger.info(
        "Chi2 Statistic: %s, p-value: %s, Degrees of Freedom: %s",
        chi2,
        p,
        dof,
    )

    chi2, p, dof, _ = chi_square_test(df["FAVC"], df["Obese_Binary"])
    logger.info("Chi-square test between FAVC and Obese_Binary:")
    logger.info(
        "Chi2 Statistic: %s, p-value: %s, Degrees of Freedom: %s",
        chi2,
        p,
        dof,
    )


if __name__ == "__main__":
    main()
