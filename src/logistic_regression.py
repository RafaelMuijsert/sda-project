"""Perform binary logistic regression analysis.

Copyright (C) 2025.
"""

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up dataset and prepare for anlysis.

    Returns:
        pd.DataFrame: the cleaned dataset.

    """
    df["obesity_binary"] = 0
    # this classifies Insufficient_Weight as obese; should update naming or check.
    df.loc[df["NObeyesdad"] != "Normal_Weight", "obesity_binary"] = 1

    df["family_history_with_overweight"] = df["family_history_with_overweight"].map(
        {"yes": 1, "no": 0},
    )
    df["SMOKE"] = df["SMOKE"].map({"yes": 1, "no": 0})
    df["FAVC"] = df["FAVC"].map({"yes": 1, "no": 0})
    df["CAEC"] = df["CAEC"].map({"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3})
    return df.dropna()


def main() -> None:
    """Read csv and perform analysis."""
    raw = pd.read_csv("data/ObesityDataSet_raw_and_data_sinthetic.csv")
    df = prepare_dataset(raw)
    x = df[["family_history_with_overweight", "SMOKE", "FAVC", "CAEC"]]
    x = sm.add_constant(x)
    y = df["obesity_binary"]
    model = sm.Logit(y, x).fit()
    plt.show()

    print(model.summary())


if __name__ == "__main__":
    main()
