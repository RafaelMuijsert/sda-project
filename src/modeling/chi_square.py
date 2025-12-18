"""Predicting obesity through hereditary and lifestyle behaviors.

Copyright (C) 2025.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def contingency_table(var1, var2) -> tuple[pd.DataFrame, pd.DataFrame]:
    ct = pd.crosstab(var1, var2)
    ct_percent = pd.DataFrame.round(ct.div(ct.sum(axis=1), axis=0) * 100, 2)

    return ct, ct_percent


def chi_square_test(var1, var2) -> tuple[float, float, int, np.ndarray]:
    ct = pd.crosstab(var1, var2)
    chi2, p, dof, ex = chi2_contingency(ct, correction=False)

    return chi2, p, dof, ex


def main() -> None:
    df = pd.read_csv("data/obesity_cleaned_final.csv", sep=";")

    chi2, p, dof, ex = chi_square_test(
        df["family_history_with_overweight"],
        df["Obese_Binary"],
    )
    print("\nChi-square test between family_history_with_overweight and Obese_Binary:")
    print(
        f"Chi2 Statistic: {chi2}, p-value: {p:.2f}, Degrees of Freedom: {dof}, Expected Frequencies:\n{ex}",
    )

    chi2, p, dof, ex = chi_square_test(df["FAVC"], df["Obese_Binary"])
    print("\nChi-square test between FAVC and Obese_Binary:") # Changed print statement to reflect FAVC
    print(
        f"Chi2 Statistic: {chi2}, p-value: {p:.2f}, Degrees of Freedom: {dof}, Expected Frequencies:\n{ex}",
    )


if __name__ == "__main__":
    main()
