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
