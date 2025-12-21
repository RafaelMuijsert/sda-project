import pandas as pd
import statsmodels.api as sm
df= pd.read_csv("data/ObesityDataSet_raw_and_data_sinthetic.csv")
df['obesity_binary']= 0
df.loc[df['NObeyesdad']!='Normal_Weight','obesity_binary']= 1
df['family_history_with_overweight']= df['family_history_with_overweight'].map({'yes': 1,'no': 0})
df['SMOKE']=df['SMOKE'].map({'yes':1,'no':0})
df['FAVC']=df['FAVC'].map({'yes': 1, 'no': 0})
df['CAEC']= df['CAEC'].map({'no': 0,'Sometimes':1,'Frequently':2,'Always':3})
df= df.dropna()
X=df[['family_history_with_overweight', 'SMOKE', 'FAVC', 'CAEC']]
X=sm.add_constantS(X)
y=df['obesity_binary']
model= sm.Logit(y, X).fit()
print(model.summary())
