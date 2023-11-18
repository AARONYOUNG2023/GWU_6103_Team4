#%% [markdown]
## Heart Risk Attack
#  This file is for EDA.

#  Remember that in this dataset, some categroy data are set as the following  .


#  'Male' to 1 and 'Female' to 0 in the 'Sex' column  .


#  'Unhealthy' to 0, 'Average' to 1, and 'Healthy' to 2 in the 'Diet' column  .

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# %%
# Importing data from CSV
df = pd.read_csv('heart_attack_prediction_dataset_revised.csv', delimiter=';', on_bad_lines = 'skip')

# Display the first few rows of the dataframe
print(df)
# %% [markdown]
## EDA steps
# 1. Understanding the Dataset
      #* View Data Samples
df.head()
#%%
df.tail()
# %% [markdown]
      #* Data Types and Missing Values
df.info()
df.isnull().sum()
# %% [markdown]
      #* Summary Statistics
df.describe()
# %% [markdown]
# 2. Cleaning the Dataset
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
# %%
# %% [markdown]
# 2. Cleaning the Dataset
     #* Converting Categorical Data
# Assuming df is your DataFrame
# Map 'Male' to 1 and 'Female' to 0 in the 'Sex' column
sex_mapping = {'Male': 1, 'Female': 0}
df['Sex'] = df['Sex'].map(sex_mapping)

# Map 'Unhealthy' to 0, 'Average' to 1, and 'Healthy' to 2 in the 'Diet' column
diet_mapping = {'Unhealthy': 0, 'Average': 1, 'Healthy': 2}
df['Diet'] = df['Diet'].map(diet_mapping)



# %% [markdown]
      #* Summary Statistics again
df.describe()
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
# %%
