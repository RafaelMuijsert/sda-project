import pandas as pd

RAW_DATA_PATH = "data/ObesityDataSet_raw_and_data_sinthetic.csv"
CLEANED_DATA_PATH = "data/obesity_cleaned_final.csv"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the raw obesity dataset.

    Args:
        df (pd.DataFrame): The raw DataFrame loaded from the CSV.

    Returns:
        pd.DataFrame: The cleaned and preprocessed DataFrame.
    """
    # Rename 'NObeyesdad' for clarity and consistency
    df = df.rename(columns={"NObeyesdad": "Obesity_Category"})

    # Map categorical features to numerical
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).astype(int)
    df["family_history_with_overweight"] = df[
        "family_history_with_overweight"
    ].map({"yes": 1, "no": 0}).astype(int)
    df["FAVC"] = df["FAVC"].map({"yes": 1, "no": 0}).astype(int)
    df["SMOKE"] = df["SMOKE"].map({"yes": 1, "no": 0}).astype(int)
    df["SCC"] = df["SCC"].map({"yes": 1, "no": 0}).astype(int)
    df["CALC"] = df["CALC"].map(
        {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    ).astype(int)
    df["CAEC"] = df["CAEC"].map(
        {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    ).astype(int)

    # One-hot encode 'MTRANS' and convert to int
    df = pd.get_dummies(df, columns=["MTRANS"], prefix="MTRANS", dtype=int)

    # Map ordinal 'Obesity_Category' to numerical 'Obesity_Level'
    obesity_level_mapping = {
        "Insufficient_Weight": 0,
        "Normal_Weight": 1,
        "Overweight_Level_I": 2,
        "Overweight_Level_II": 3,
        "Obesity_Type_I": 4,
        "Obesity_Type_II": 5,
        "Obesity_Type_III": 6,
    }
    df["Obesity_Level"] = df["Obesity_Category"].map(obesity_level_mapping)

    # Create 'Obese_Binary' column (1 if obese/overweight, 0 otherwise)
    # Fix: Only classify 'Normal_Weight' and 'Insufficient_Weight' as 0
    df["Obese_Binary"] = df["Obesity_Category"].apply(
        lambda x: 0
        if x in ["Normal_Weight", "Insufficient_Weight"]
        else 1,
    )

    # Handle missing values (if any) - drop rows with any NaN values
    df = df.dropna()

    return df


def main() -> None:
    """Load raw data, clean it, and save the cleaned data."""
    raw_df = pd.read_csv(RAW_DATA_PATH)
    cleaned_df = clean_data(raw_df)
    cleaned_df.to_csv(CLEANED_DATA_PATH, sep=";", index=False)
    print(f"Cleaned data saved to {CLEANED_DATA_PATH}")  # noqa: T201


if __name__ == "__main__":
    main()
