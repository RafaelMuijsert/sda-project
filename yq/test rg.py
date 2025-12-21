import pandas as pd
import statsmodels.api as sm
import os
script_dir= os.path.dirname(os.path.abspath(__file__))
project_root=os.path.abspath(os.path.join(script_dir, ".."))
data_path= os.path.join(project_root,"data","ObesityDataSet_raw_and_data_sinthetic.csv")
df= pd.read_csv(data_path)
df['obesity_binary']= 0
df.loc[df['NObeyesdad']!='Normal_Weight','obesity_binary']= 1
df['family_history_with_overweight']= df['family_history_with_overweight'].map({'yes': 1,'no': 0})
df['SMOKE']=df['SMOKE'].map({'yes':1,'no':0})
df['FAVC']=df['FAVC'].map({'yes': 1, 'no': 0})
df['CAEC']= df['CAEC'].map({'no': 0,'Sometimes':1,'Frequently':2,'Always':3})
df= df.dropna()
X=df[['family_history_with_overweight', 'SMOKE', 'FAVC', 'CAEC']]
X=sm.add_constant(X)
y=df['obesity_binary']
model= sm.Logit(y, X).fit()
print(model.summary())
