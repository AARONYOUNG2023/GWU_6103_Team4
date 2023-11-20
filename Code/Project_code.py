#%% [markdown]
## Heart Attack Prediction
## For this project, we used a redesigned dataset in which we took the important columns relevant to heart attack prediction while removing the unnecessary columns from the original dataset.


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import os

# %%
# Importing dataset from CSV
df = pd.read_csv('heart_attack_prediction_dataset_revised.csv', delimiter=';', on_bad_lines = 'skip')
df

# %%
# Displaying the first few rows of the dataset
print(df.head())


#%%
# Summary statistics of numerical features
print(df.describe())


#%%
# Data set cleaning
print("Missing Values:")
print(df.isnull().sum())

# Drop rows with missing values (if necessary)
df = df.dropna()


# Display cleaned dataset
print("Cleaned Dataset:")
print(df.head())
#%%

print("Column Names:", df.columns)


# %% [markdown]
#  Cleaning the Dataset
     #* Cleaning Concatenated Numbers
def clean_concatenated_numbers(column):
    cleaned = []
    for value in df[column]:
        # Assuming the numbers are concatenated without any separator
        digits = [int(digit) for digit in str(value) if digit.isdigit()]
        if digits:
            # Replace this with any other method you deem suitable
            average_value = sum(digits) / len(digits)
            cleaned.append(average_value)
        else:
            cleaned.append(None)
    return cleaned

df['Exercise Hours Per Week'] = clean_concatenated_numbers('Exercise Hours Per Week')
df['Sedentary Hours Per Day'] = clean_concatenated_numbers('Sedentary Hours Per Day')

# %% [markdown]
#  Cleaning the Dataset
     #* Converting Categorical Data
# Assuming df is your DataFrame
# Map 'Male' to 1 and 'Female' to 0 in the 'Sex' column
sex_mapping = {'Male': 1, 'Female': 0}
df['Sex'] = df['Sex'].map(sex_mapping)

# Map 'Unhealthy' to 0, 'Average' to 1, and 'Healthy' to 2 in the 'Diet' column
diet_mapping = {'Unhealthy': 0, 'Average': 1, 'Healthy': 2}
df['Diet'] = df['Diet'].map(diet_mapping)

# %% [markdown]
# 3. Histograms for Numeric Variables

# List of numerical variables
numerical_vars = ['Age', 'Cholesterol', 'Heart Rate', 'Exercise Hours Per Week', 
                  'Sedentary Hours Per Day', 'Physical Activity Days Per Week', 
                  'Sleep Hours Per Day']

# Number of rows for subplots
n_rows_num = len(numerical_vars)

# Set up the matplotlib figure for numerical variables
fig, axes = plt.subplots(n_rows_num, 1, figsize=(8, 4 * n_rows_num))

# If only one numerical column, axes might not be an array
if n_rows_num == 1:
    axes = [axes]

# Loop through the numerical variables and create a histogram for each
for i, col in enumerate(numerical_vars):
    df[col].hist(ax=axes[i], bins=20, edgecolor='black')
    axes[i].set_title(f'Histogram of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# %% [markdown]
# 4. Bar Charts for Categorical Variables
# Exclude the numerical variables and identifiers to get the categorical ones
excluded_vars = numerical_vars + ['Patient ID']  # Add any other columns you wish to exclude
categorical_vars = df.columns.difference(excluded_vars)

# Number of rows for subplots
n_rows_cat = len(categorical_vars)

# Set up the matplotlib figure for categorical variables
fig, axes = plt.subplots(n_rows_cat, 1, figsize=(8, 4 * n_rows_cat))

# If only one categorical column, axes might not be an array
if n_rows_cat == 1:
    axes = [axes]

# Loop through the categorical variables and create a bar chart for each
for i, col in enumerate(categorical_vars):
    sns.countplot(x=col, data=df, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel('Category')
    axes[i].set_ylabel('Count')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


#%%
df_numeric = df.select_dtypes(include=['float64', 'int64'])
#This code used select_dtypes to include only numeric columns in the correlation matrix computation.

# Calculate the correlation matrix
matrix_correlation = df_numeric.corr()

# Use Seaborn's heatmap to plot the correlation matrix.
plt.figure(figsize=(10, 8))
sns.heatmap(matrix_correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


# %%
#In this dataset, find the values for the factors that are strongest predictors of heart attack risk, in decreasing order.



df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
matrix_correlation = df_numeric.corr()

# Determine the relationship between each characteristic and the target variable.

target_correlation = matrix_correlation['Heart Attack Risk'].abs()

# Sort the characteristics according to their relationship to the target variable.

sorted_correlation = target_correlation.sort_values(ascending=False)

# Print the features with the highest correlation coefficients
print("Features with the highest correlation in descending order:")
print(sorted_correlation)


#%%
# Choose the top five correlated features. 
top5_features = sorted_correlation.index[1:6]

# Display the top 5 correlated features
print("Top 5 Correlated Features with Heart Attack Risk:")
print(top5_features)


# %%
# Filter rows where "Heart Attack Risk" is 1
heart_attack_df = df[df['Heart Attack Risk'] == 1]

# Select top 5 features correlated with "Heart Attack Risk"
top5_features = sorted_correlation.index[1:6]  # Exclude 'Heart Attack Risk' itself

# Plot histograms for the top 5 correlated features
for feature in top5_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(heart_attack_df[feature], bins=20, kde=True,color='red')
    plt.title(f'Histogram of {feature} for Heart Attack Risk = 1')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


# %%
#Relation betwwen cholestrol and heart attack risk
plt.figure(figsize=(10, 8))
sns.regplot(x='Cholesterol', y='Heart Attack Risk', data=df, logistic=True, scatter_kws={'s': 20},color='pink')
plt.title('Relationship Between Cholesterol Levels and Heart Attack Risk')
plt.xlabel('Cholesterol Levels')
plt.ylabel('Heart Attack Risk')
plt.show()

#Because the line is slanted upward, it has a positive slope, indicating that greater cholesterol levels are connected with an increased risk of a heart attack.
# %%

# Calculate the correlation coefficient
cholesterol_heart_attack_correlation = df['Cholesterol'].corr(df['Heart Attack Risk'])

# Print the correlation coefficient
print(f"Correlation between Cholesterol and Heart Attack Risk: {cholesterol_heart_attack_correlation}")


# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='Heart Attack Risk', y='Cholesterol', data=heart_attack_df, color='purple')
plt.title('Box Plot of Cholesterol for Heart Attack Risk = 1')
plt.xlabel('Heart Attack Risk')
plt.ylabel('Cholesterol')
plt.show()

#This code calculates the correlation coefficient between 'Cholesterol' and 'Heart Attack Risk' and then creates a box plot to visualize the distribution of cholesterol levels in relation to heart attack risk

# %%
# Define age groups
age_bins = [0, 29, 39, 49, 59, 69, 79, 89, 99]
age_labels = ['0-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Create a contingency table for obesity and diabetes for the entire dataset
contingency_table_all = pd.crosstab(df['Obesity'], df['Diabetes'])

# Plotting the contingency table as a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(contingency_table_all, annot=True, fmt='d', cmap="YlGnBu")
plt.title('Heatmap of Diabetes Prevalence vs. Obesity for Entire Dataset')
plt.xlabel('Diabetes')
plt.ylabel('Obesity')
plt.show()

# Calculate proportions of diabetes for obese and non-obese in each age group
age_group_proportions = df.groupby('Age Group').apply(
    lambda x: pd.Series({
        'Proportion with Diabetes (Obese)': x[x['Obesity'] == 1]['Diabetes'].mean(),
        'Proportion with Diabetes (Non-Obese)': x[x['Obesity'] == 0]['Diabetes'].mean()
    })
).reset_index()

# Plotting the proportions as a bar chart
plt.figure(figsize=(14, 7))
age_group_proportions.set_index('Age Group').plot(kind='bar', stacked=False)
plt.title('Proportion of Diabetes in Obese vs Non-Obese Patients Across Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Proportion with Diabetes')
plt.xticks(rotation=45)
plt.legend(title='Obesity Status')
plt.tight_layout()
plt.show()

# %%
## SMART QUESTION: Is there a statistically significant difference in the prevalence of diabetes between obese and non-obese patients, and does this difference vary by age group?
from scipy.stats import chi2_contingency

## Define age groups
age_bins = [0, 29, 39, 49, 59, 69, 79, 89, 99]
age_labels = ['0-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Create a contingency table for obesity and diabetes for the entire dataset
contingency_table_all = pd.crosstab(df['Obesity'], df['Diabetes'])

# Perform the chi-squared test for the entire dataset
chi2, p_value_all, dof, expected = chi2_contingency(contingency_table_all)

# Perform the chi-squared test for each age group and store results
age_group_results = []
for group in age_labels:
    age_group_data = df[df['Age Group'] == group]
    contingency_table_age_group = pd.crosstab(age_group_data['Obesity'], age_group_data['Diabetes'])
    chi2_age, p_value_age, dof_age, expected_age = chi2_contingency(contingency_table_age_group)
    age_group_results.append((group, chi2_age, p_value_age))

# Display results
contingency_table_all, chi2, p_value_all, age_group_results


#In all age groups, the p-values are greater than 0.05, indicating that there is no statistically significant association between obesity and diabetes within these specific age categories. However, the '60-69' age group shows a p-value closest to 0.05, suggesting a potential trend that might be worth exploring further, especially in a larger dataset or a more focused study.
# %%
## SMART QUESTION: How do age and gender interact as determinants of heart attack risk and are there age specifics or gender-specifics patterns?
# Analyzing the interaction between age, gender, and heart attack risk
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
age_gender_risk = df.groupby(['Age', 'Sex'])['HeartAttackRisk'].mean().reset_index()

# Line Plot
plt.figure(figsize=(15, 8))
sns.lineplot(data=age_gender_risk, x='Age', y='HeartAttackRisk', hue='Sex')
plt.title('Heart Attack Risk by Age and Gender')
plt.xlabel('Age')
plt.ylabel('Average Heart Attack Risk')
plt.legend(title='Gender')
plt.show()

# Boxplot
plt.figure(figsize=(15, 8))
sns.boxplot(x='Age', y='HeartAttackRisk', hue='Sex', data=df)
plt.title('Distribution of Heart Attack Risk by Age and Gender')
plt.xlabel('Age')
plt.ylabel('Heart Attack Risk')
plt.xticks(rotation=45)
plt.show()

# Statistical Testing: Two-Way ANOVA
model = ols('HeartAttackRisk ~ C(Age) + C(Sex)', data=df).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)

# %%
