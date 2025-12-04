from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np

# Verify the file path is correct
df = pd.read_csv('obesity_cleaned_final.csv')

def contingency_table(var1, var2) -> tuple[pd.DataFrame, pd.DataFrame]:
    ct = pd.crosstab(var1, var2)
    ct_percent = pd.DataFrame.round(ct.div(ct.sum(axis=1), axis=0) * 100,2)
    
    return ct, ct_percent

def chi_square_test(var1, var2) -> tuple[float, float, int, np.ndarray]:
    ct = pd.crosstab(var1, var2)
    chi2, p, dof, ex = chi2_contingency(ct, correction=False)
    
    return chi2, p, dof, ex

# usage chi_square_test
chi2, p, dof, ex = chi_square_test(df['family_history_with_overweight'], df['Obese_Binary'])
print(f"\nChi-square test between family_history_with_overweight and Obese_Binary:")
print(f"Chi2 Statistic: {chi2}, p-value: {p:.2f}, Degrees of Freedom: {dof}, Expected Frequencies:\n{ex}")

chi2, p, dof, ex = chi_square_test(df['FAVC'], df['Obese_Binary'])
print(f"\nChi-square test between family_history_with_overweight and Obese_Binary:")
print(f"Chi2 Statistic: {chi2}, p-value: {p:.2f}, Degrees of Freedom: {dof}, Expected Frequencies:\n{ex}")
