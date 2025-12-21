import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

'''
    Gender: Feature, Categorical, "Gender"
    Age : Feature, Continuous, "Age"
    Height: Feature, Continuous
    Weight: Feature Continuous
    family_history_with_overweight: Feature, Binary, " Has a family member suffered or suffers from overweight? "

    FAVC : Feature, Binary, " Do you eat high caloric food frequently? "
    FCVC : Feature, Integer, " Do you usually eat vegetables in your meals? "
    NCP : Feature, Continuous, " How many main meals do you have daily? "
    CAEC : Feature, Categorical, " Do you eat any food between meals? "
    SMOKE : Feature, Binary, " Do you smoke? "
    CH2O: Feature, Continuous, " How much water do you drink daily? "
    SCC: Feature, Binary, " Do you monitor the calories you eat daily? "
    FAF: Feature, Continuous, " How often do you have physical activity? "
    TUE : Feature, Integer, " How much time do you use technological devices such as cell phone, videogames, television, computer and others? "

    CALC : Feature, Categorical, " How often do you drink alcohol? "
    MTRANS : Feature, Categorical, " Which transportation do you usually use? "
    NObeyesdad : Target, Categorical, "Obesity level"
    
'''

# Load the data
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
print(f"Original shape: {df.shape}")

# Remove duplicates + some rounding
df = df.drop_duplicates().reset_index(drop=True)
df['Age']    = df['Age'].round(0).astype(int)
df['Height'] = df['Height'].round(4)
df['Weight'] = df['Weight'].round(3)
df['CH2O']   = df['CH2O'].round(3)
df['FAF']    = df['FAF'].round(3)
df['TUE']    = df['TUE'].round(3)

print(f"After deduplication: {df.shape}")

# Encode into binary = yes/no to 1/0
binary_cols = ['FAVC', 'SMOKE', 'SCC', 'family_history_with_overweight']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0}).astype('int64')

# Keep column name "Gender", but make numeric (1 = male, 0 = female)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).astype('int64')

# CAEC & CALC to ordinal encoding
df['CAEC'] = df['CAEC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}).astype('int64')
df['CALC'] = df['CALC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}).astype('int64')

# MTRANS to dummies
mtrans_dummies = pd.get_dummies(df['MTRANS'], prefix='MTRANS', dtype=int)
df = pd.concat([df.drop('MTRANS', axis=1), mtrans_dummies], axis=1)

# Target variables
obesity_order = {
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6
}
df['Obesity_Level'] = df['NObeyesdad'].map(obesity_order).astype('int64')
df['Obese_Binary']  = (df['Obesity_Level'] >= 4).astype('int64')

# Final column order
final_columns = [
    'Gender', 'Age', 'Height', 'Weight',
    'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP',
    'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC',
    'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
    'MTRANS_Public_Transportation', 'MTRANS_Walking',
    'NObeyesdad',          
    'Obesity_Level',       
    'Obese_Binary'          
]

# Add any missing MTRANS columns with zeros
for col in ['MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
            'MTRANS_Public_Transportation', 'MTRANS_Walking']:
    if col not in df.columns:
        df[col] = 0

clean_df = df[final_columns].copy()
print(f"Total missing values: {clean_df.isnull().sum().sum()} (should be 0)")

'''
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 12))

ax1 = plt.subplot(2, 3, 1)
order = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
df['NObeyesdad'].value_counts().reindex(order).plot(kind='bar', ax=ax1)
ax1.set_title('Obesity Level Distribution', fontsize=14, fontweight='bold')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)

ax2 = plt.subplot(2, 3, 2)
ct = pd.crosstab(df['family_history_with_overweight'], df['Obese_Binary'])
ct.columns = ['Not Obese', 'Obese']
ct.index = ['No', 'Yes']
ct.plot(kind='bar', stacked=True, ax=ax2)
ax2.set_title('Obesity by Family History', fontsize=14, fontweight='bold')
ax2.set_xlabel('Family History of Overweight')
ax2.set_ylabel('Count')
ax2.legend(title='Obesity Status')

ax3 = plt.subplot(2, 3, 3)
scatter = ax3.scatter(df['Age'], df['Weight'], c=df['Obesity_Level'], cmap='viridis', alpha=0.7, s=50)
plt.colorbar(scatter, ax=ax3, label='Obesity Level (0–6)')
ax3.set_title('Age vs Weight by Obesity Level', fontsize=14, fontweight='bold')
ax3.set_xlabel('Age')
ax3.set_ylabel('Weight (kg)')

ax4 = plt.subplot(2, 3, 4)
sns.boxplot(x='Obese_Binary', y='FAF', data=df, ax=ax4)
ax4.set_xticklabels(['Not Obese', 'Obese'])
ax4.set_title('Physical Activity Frequency (FAF)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Obesity Status')
ax4.set_ylabel('FAF (0–3)')

ax5 = plt.subplot(2, 3, 5)
pd.crosstab(df['FAVC'], df['Obese_Binary']).plot(kind='bar', stacked=True, ax=ax5)
ax5.set_xticklabels(['Rarely', 'Frequently'], rotation=0)
ax5.set_title('High-Caloric Food Intake (FAVC)', fontsize=14, fontweight='bold')
ax5.set_xlabel('Eats High-Caloric Food Frequently')
ax5.set_ylabel('Count')
ax5.legend(['Not Obese', 'Obese'], title='Status')

ax6 = plt.subplot(2, 3, 6)
mtrans_cols = [c for c in df.columns if c.startswith('MTRANS_')]
counts = df[mtrans_cols].sum().sort_values(ascending=False)
labels = ['Public Transport', 'Walking', 'Automobile', 'Bike', 'Motorbike']
counts.reindex(['MTRANS_Public_Transportation', 'MTRANS_Walking','MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike']).plot(kind='bar', ax=ax6)
ax6.set_title('Transportation Mode', fontsize=14, fontweight='bold')
ax6.set_ylabel('Count')
ax6.set_xticklabels(labels, rotation=30)

plt.suptitle('Obesity Dataset - Key Exploratory Visualizations', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
'''