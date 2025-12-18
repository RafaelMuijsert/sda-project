"""Generate boxplots of lifestyle variables vs. obesity category."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.ion()
df = pd.read_csv("data/obesity_cleaned_final.csv", sep=";")

lifestyle_vars = ["FAF", "FCVC", "TUE", "CALC", "CH2O"]
titles = {
    "FAF": "Physical activity",
    "FCVC": "Vegetable intake",
    "TUE": "Screen time",
    "CALC": "Sugary drinks",
    "CH2O": "Water intake",
}

for var in lifestyle_vars:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Obesity_Category", y=var, data=df)
    plt.title(f"{titles[var]} vs Obesity Category")
    plt.xlabel("Obesity Category")
    plt.ylabel(titles[var])
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"results/lifestyle_{var}.png")
