"""Predicting obesity through hereditary and lifestyle behaviors.

Copyright (C) 2025.
"""

import kagglehub
import pandas as pd
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np

def load_dataset() -> pd.DataFrame:
    """Load the dataset from kaggle."""
    # Set the path to the file you'd like to load
    file_path = (
        "estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.csv"
    )

    # Load the latest version
    df: pd.DataFrame = kagglehub.dataset_load(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType, reportUnknownMemberType]
        kagglehub.KaggleDatasetAdapter.PANDAS,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        "ikjotsingh221/obesity-risk-prediction-cleaned",
        file_path,
    )
    
    # Remove duplicates + some rounding
    df = df.drop_duplicates().reset_index(drop=True)
    df['Age']    = df['Age'].round(0).astype(int)
    df['Height'] = df['Height'].round(4)
    df['Weight'] = df['Weight'].round(3)
    df['CH2O']   = df['CH2O'].round(3)
    df['FAF']    = df['FAF'].round(3)
    df['TUE']    = df['TUE'].round(3)

    print(f"After deduplication: {df.shape}")

    # Encode into binary = yes/no to 1/0
    binary_cols = ['FAVC', 'SMOKE', 'SCC', 'family_history_with_overweight']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0}).astype('int64')

    # Keep column name "Gender", but make numeric (1 = male, 0 = female)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).astype('int64')

    # CAEC & CALC to ordinal encoding
    df['CAEC'] = df['CAEC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}).astype('int64')
    df['CALC'] = df['CALC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}).astype('int64')

    # MTRANS to dummies
    mtrans_dummies = pd.get_dummies(df['MTRANS'], prefix='MTRANS', dtype=int)
    df = pd.concat([df.drop('MTRANS', axis=1), mtrans_dummies], axis=1)

    # Target variables
    obesity_order = {
        'Insufficient_Weight': 0,
        'Normal_Weight': 1,
        'Overweight_Level_I': 2,
        'Overweight_Level_II': 3,
        'Obesity_Type_I': 4,
        'Obesity_Type_II': 5,
        'Obesity_Type_III': 6
    }
    df['Obesity_Level'] = df['NObeyesdad'].map(obesity_order).astype('int64')
    df['Obese_Binary']  = (df['Obesity_Level'] >= 4).astype('int64')

    # Final column order
    final_columns = [
        'Gender', 'Age', 'Height', 'Weight',
        'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP',
        'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC',
        'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
        'MTRANS_Public_Transportation', 'MTRANS_Walking',
        'NObeyesdad',          
        'Obesity_Level',       
        'Obese_Binary'          
    ]

    # Add any missing MTRANS columns with zeros
    for col in ['MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
                'MTRANS_Public_Transportation', 'MTRANS_Walking']:
        if col not in df.columns:
            df[col] = 0

    clean_df = df[final_columns].copy()
    
    return df

df = load_dataset()

def contingency_table(var1, var2) -> tuple[pd.DataFrame, pd.DataFrame]:
    ct = pd.crosstab(var1, var2)
    ct_percent = pd.DataFrame.round(ct.div(ct.sum(axis=1), axis=0) * 100,2)
    
    return ct, ct_percent

def chi_square_test(var1, var2) -> tuple[float, float, int, np.ndarray]:
    ct = pd.crosstab(var1, var2)
    chi2, p, dof, ex = chi2_contingency(ct, correction=False)
    
    return chi2, p, dof, ex

def main() -> None:
    df = load_dataset()
    chi2, p, dof, ex = chi_square_test(df['family_history_with_overweight'], df['Obese_Binary'])
    print(f"\nChi-square test between family_history_with_overweight and Obese_Binary:")
    print(f"Chi2 Statistic: {chi2}, p-value: {p:.2f}, Degrees of Freedom: {dof}, Expected Frequencies:\n{ex}")

    chi2, p, dof, ex = chi_square_test(df['FAVC'], df['Obese_Binary'])
    print(f"\nChi-square test between family_history_with_overweight and Obese_Binary:")
    print(f"Chi2 Statistic: {chi2}, p-value: {p:.2f}, Degrees of Freedom: {dof}, Expected Frequencies:\n{ex}")



if __name__ == "__main__":
    main()
