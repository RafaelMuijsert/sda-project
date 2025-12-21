import matplotlib.pyplot as plt
import pandas as pd
data= {"variable": ["Family history","SMOKE","FAVC","CAEC"],"coef": [1.3358,-0.9382,0.7846,-1.1108],"ci_lower": [1.042,-1.667,0.437,-1.340],"ci_upper": [1.629,-0.209,1.132,-0.881]}
df=pd.DataFrame(data)
df["abs_coef"]=df["coef"].abs()
df =df.sort_values("abs_coef", ascending=False)
plt.figure()
plt.errorbar(df["coef"],df["variable"],xerr=[df["coef"] - df["ci_lower"], df["ci_upper"] - df["coef"]],fmt="o")
plt.axvline(0)
plt.gca().invert_yaxis()
plt.xlabel("Log-odds (β)")
plt.title("Logistic Regression Coefficient Plot")
plt.tight_layout()
plt.show()
