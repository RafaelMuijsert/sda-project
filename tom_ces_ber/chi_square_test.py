import pandas as pd
import numpy as np
import matplotlib.pyplot as plt2
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu

df = pd.read_csv('obesity_cleaned_final.csv')

target = 'Obese_Binary'  # 1 = Obese (Types I-III), 0 = Non-obese

continuous_vars = ['CH2O', 'FAF', 'TUE']   
binary_var = 'SCC'                         # Binary → use Chi-Square test

results = []

print("BIVARIATE TESTS FOR SPECIFIED LIFESTYLE VARIABLES vs Obesity (Binary)\n")
print("="*80)

for var in continuous_vars:
    if var not in df.columns:
        print(f"Warning: {var} not in dataset → skipped")
        continue
    
    group_nonobese = df[df[target] == 0][var]
    group_obese = df[df[target] == 1][var]
    
    stat, p = mannwhitneyu(group_nonobese, group_obese, alternative='two-sided')
    
    mean_nonobese = group_nonobese.mean()
    median_nonobese = group_nonobese.median()
    mean_obese = group_obese.mean()
    median_obese = group_obese.median()
    
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    
    results.append({
        'Variable': var,
        'Test': 'Mann-Whitney U',
        'Statistic': round(stat, 3),
        'p-value': p,
        'p_formatted': f"{p:.3e}" if p < 0.001 else f"{p:.4f}",
        'Signif.': sig,
        'Mean (Non-Obese)': round(mean_nonobese, 3),
        'Mean (Obese)': round(mean_obese, 3),
        'Median (Non-Obese)': round(median_nonobese, 3),
        'Median (Obese)': round(median_obese, 3)
    })
    
    print(f"\n→ {var} vs {target} (Mann-Whitney U Test)")
    print(f"Non-Obese (n={len(group_nonobese)}): Mean = {mean_nonobese:.3f}, Median = {median_nonobese:.3f}")
    print(f"Obese     (n={len(group_obese)}): Mean = {mean_obese:.3f}, Median = {median_obese:.3f}")
    print(f"U = {stat:.3f}, p = {p:.3e} {sig}")
    print("-" * 80)

if binary_var in df.columns:
    counts = pd.crosstab(df[binary_var], df[target])
    percentages = pd.crosstab(df[binary_var], df[target], normalize='index') * 100
    percentages = percentages.round(2)
    
    chi2, p, dof, expected = chi2_contingency(counts)
    
    min_expected = np.min(expected)
    warning = " (Caution: some expected <5)" if min_expected < 5 else ""
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    
    results.append({
        'Variable': binary_var,
        'Test': 'Chi-Square',
        'Statistic': round(chi2, 3),
        'df': dof,
        'p-value': p,
        'p_formatted': f"{p:.3e}" if p < 0.001 else f"{p:.4f}",
        'Signif.': sig,
        'Min Expected': round(min_expected, 2)
    })
    
    print(f"\n→ {binary_var} vs {target} (Chi-Square Test)")
    print("Counts:")
    print(counts)
    print("\nRow Percentages (% Obese):")
    print(percentages.iloc[:, 1] if 1 in percentages.columns else percentages.iloc[:, 0])
    print(f"\nChi² = {chi2:.3f}, df = {dof}, p = {p:.3e}{warning} {sig}")
    print("-" * 80)
else:
    print(f"Warning: {binary_var} not in dataset → skipped")

# --- Summary table ---
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('p-value')

print("\nSUMMARY OF BIVARIATE TESTS")
print("="*80)
if 'df' in results_df.columns:
    print(results_df[['Variable', 'Test', 'Statistic', 'p_formatted', 'Signif.', 'Mean (Non-Obese)', 'Mean (Obese)']].to_string(index=False))
else:
    print(results_df[['Variable', 'Test', 'Statistic', 'p_formatted', 'Signif.']].to_string(index=False))

# Save results
results_df.to_csv('lifestyle_bivariate_results.csv', index=False)
print("\nResults saved to 'lifestyle_bivariate_results.csv'")

print("\nAnalysis complete. Use these results/plots for the bivariate associations of the key lifestyle factors.")

import matplotlib.pyplot as plt
import seaborn as sns

# Continuous variables for plotting
continuous_vars = ['CH2O', 'FAF', 'TUE']
colors = ['skyblue', 'salmon']  # Non-Obese, Obese

# Create one big figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18,6), sharey=False)

for i, var in enumerate(continuous_vars):
    ax = axes[i]
    sns.boxplot(x=target, y=var, data=df, palette=colors, ax=ax)
    ax.set_xticklabels(['Non-Obese', 'Obese'])
    ax.set_title(f"{var} vs Obesity Status")
    ax.set_xlabel("")
    ax.set_ylabel(var)

    # Annotate median values
    med_nonobese = df[df[target]==0][var].median()
    med_obese = df[df[target]==1][var].median()
    ax.text(-0.1, med_nonobese, f"{med_nonobese:.2f}", color='blue', fontweight='bold')
    ax.text(0.9, med_obese, f"{med_obese:.2f}", color='red', fontweight='bold')

# Add overall figure title
fig.suptitle("Bivariate Distribution of Continuous Lifestyle Variables by Obesity Status", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
plt.show()
