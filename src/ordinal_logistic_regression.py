import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

DATASET_PATH = "data/obesity_cleaned_final.csv"


def read_data(filename):
    df = pd.read_csv(filename, sep=";")
    return df


def prepare_for_ordinal_regression(df):
    # Target (ordinal)
    y = df["Obesity_Level"]

    # Predictors
    X = df[
        [
            "Gender",
            "Age",
            "CALC",
            "MTRANS_Automobile",
            "MTRANS_Bike",
            "MTRANS_Motorbike",
            "MTRANS_Public_Transportation",
            # MTRANS_Walking is DROPPED (baseline) because including it made the sum of MTRANS=1
        ]
    ]

    return X, y


def run_ordinal_logistic_regression(X, y):
    model = OrderedModel(y, X, distr="logit")

    result = model.fit(method="bfgs")
    return result


def main():
    df = read_data(DATASET_PATH)

    X, y = prepare_for_ordinal_regression(df)

    result = run_ordinal_logistic_regression(X, y)

    print(result.summary())


if __name__ == "__main__":
    main()
