import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
desktop= os.path.join(os.path.expanduser("~"), "Desktop")
plt.ion()
df=pd.read_csv("../data/ObesityDataSet_raw_and_data_sinthetic.csv")
df=df.rename(columns={'NObeyesdad': 'obesity_level'})
lifestyle_vars=['FAF','FCVC','TUE','CALC','CH2O']
titles= {'FAF':'Physical activity','FCVC':'Vegetable intake','TUE': 'Screen time','CALC':'Sugary drinks','CH2O':'Water intake'}
for var in lifestyle_vars:
    plt.figure(figsize=(10,6))
    sns.boxplot(x='obesity_level',y=var,data=df)
    plt.title(f"{titles[var]} vs Obesity level")
    plt.xlabel("Obesity level")
    plt.ylabel(titles[var])
    plt.tight_layout()
    os.makedirs("yq/figures", exist_ok=True)
    plt.savefig(os.path.join(desktop, f"{var}_vs_obesity.png"),dpi=300)
    plt.show()
