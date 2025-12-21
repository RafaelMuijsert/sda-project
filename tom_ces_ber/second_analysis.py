import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("obesity_cleaned_final.csv")

y = df["Obesity_Level"]

lifestyle_vars = ["CH2O", "FAF", "TUE", "SCC"]
family_var = ["family_history_with_overweight"]
controls = ["Gender", "Age"]

X = df[lifestyle_vars + family_var + controls]

# Standardize predictors to compare effect sizes
scaler = StandardScaler()
X_std = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Fit ordinal logistic regression
model = OrderedModel(y, X_std, distr="logit")
res = model.fit(method="bfgs")

print(res.summary())

# Odds ratios
odds_ratios = np.exp(res.params)
print("\nOdds ratios:\n", odds_ratios)

# Joint Wald test: lifestyle vs family history
R = np.zeros((len(lifestyle_vars), len(res.params)))
for i, var in enumerate(lifestyle_vars):
    R[i, res.params.index.get_loc(var)] = 1

wald_lifestyle = res.wald_test(R)
print("\nJoint Wald test (lifestyle variables):")
print(wald_lifestyle)

# Single Wald test: family history
R_fam = np.zeros((1, len(res.params)))
R_fam[0, res.params.index.get_loc("family_history_with_overweight")] = 1

wald_family = res.wald_test(R_fam)
print("\nWald test (family history):")
print(wald_family)
