import pandas as pd
import matplotlib.pyplot as plt
import os
df= pd.read_csv("../data/ObesityDataSet_raw_and_data_sinthetic.csv")
df= df.rename(columns={'NObeyesdad': 'obesity level'})
crosstab_table= pd.crosstab(df['family_history_with_overweight'],df['obesity level'])
crosstab_table.index= ['No family history', 'Has family history']
crosstab_table.columns= ['less weight', 'Normal weight', 'Overweight I', 'Overweight II','Obesity I', 'Obesity II', 'Obesity III']
print(crosstab_table)
ax= crosstab_table.T.plot(kind='bar')
ax.set_xlabel("Obesity level")
ax.set_ylabel("Number of participants")
ax.set_title("Obesity vs. family history of overweight")
plt.xticks(rotation=20)
plt.tight_layout()
os.makedirs("yq/figures", exist_ok=True)
plt.savefig("yq/figures/family_history_vs_obesity.png",dpi=300)
plt.show()
