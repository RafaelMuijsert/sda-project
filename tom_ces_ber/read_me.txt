READ ME - TOMMASO BERTONASCO ,  14885808

-------- clean_dataset.py --------
The raw Obesity Dataset is loaded and cleaned (duplicates are removed, numeric values are rounded, categorical/binary variables are encoded) using this script.  

The useful target columns are created: ordered Obesity_Level (0-6) and binary Obese_Binary (obese vs non-obese).  

A clean, analysis-ready dataframe, which is saved as columns in final_columns (ready for modeling or further EDA), is prepared.

-------- chi_square_test.py -------- 
The code is designed to conduct bivariate tests of significance for the analysis of a healthy lifestyle (daily water intake CH2O, physical activity frequency FAF, tech device time TUE, and calorie monitoring SCC) against obesity status (Obese_Binary) using the preprocessed dataset.

To obtain the results, the Mann-Whitney U tests are applied for the three continuous variables and the Chi-Square test is done for the binary SCC variable which, in turn, returns the detailed outputs of the tests (means, medians, p-values, significance stars) and a sorted summary table is saved to 'lifestyle_bivariate_results.csv'.

At last, the code outputs a three paneled boxplot figure which displays the distribution of CH2O, FAF, and TUE categorized by obesity status (with median values marked) thereby providing a quick visual comparison of lifestyle differences between non-obese and obese groups.

-------- second_analysis.py -------- 
The code employs the proportional odds model (also known as the ordinal logistic regression) to fit the cleaned dataset with an ordered Obesity_Level (0–6) as an outcome and the selected lifestyle factors (CH2O, FAF, TUE, and SCC), family history, and controls (Gender, Age) as predictors.

The standardized predictors allow for a fair comparison of effect sizes; the complete model summary, exponentiated coefficients (odds ratios), a simultaneous Wald test for the four lifestyle variables, and a separate Wald test for family history are printed.

The output facilitates the assessment and ranking of the collective and individual impacts of the modifiable lifestyle factors compared to family history in the context of multi-level obesity severity.
