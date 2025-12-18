"""Generate a plot of obesity vs. family history of overweight."""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/obesity_cleaned_final.csv", sep=";")
crosstab_table = pd.crosstab(
    df["family_history_with_overweight"],
    df["Obesity_Category"],
)
crosstab_table.index = ["No family history", "Has family history"]
crosstab_table.columns = [
    "Insufficient Weight",
    "Normal Weight",
    "Overweight I",
    "Overweight II",
    "Obesity I",
    "Obesity II",
    "Obesity III",
]

ax = crosstab_table.T.plot(kind="bar")
ax.set_xlabel("Obesity Level")
ax.set_ylabel("Number of participants")
ax.set_title("Obesity vs. family history of overweight")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("results/family_history_vs_obesity.png")
