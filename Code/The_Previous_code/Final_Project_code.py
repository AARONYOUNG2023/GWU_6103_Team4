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


# In all age groups, the p-values are greater 
# than 0.05, indicating that there is no 
# statistically significant association 
# between obesity and diabetes within these
# specific age categories. However, the 
# '60-69' age group shows a p-value closest 
# to 0.05, suggesting a potential trend that
# might be worth exploring further, 
# especially in a larger dataset or 
# a more focused study.

# %%
## SMART QUESTION: How do age and gender interact as determinants of heart attack risk and are there age specifics or gender-specifics patterns?
# Analyzing the interaction between age, gender, and heart attack risk
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
age_gender_risk = df.groupby(['Age', 'Sex'])['Heart Attack Risk'].mean().reset_index()

#%%[markdown]
# Line Plot
plt.figure(figsize=(15, 8))
sns.lineplot(data=age_gender_risk, x='Age', y='Heart Attack Risk', hue='Sex')
plt.title('Heart Attack Risk by Age and Gender')
plt.xlabel('Age')
plt.ylabel('Average Heart Attack Risk')
plt.legend(title='Gender')
plt.show()

#%%
# Boxplot
plt.figure(figsize=(15, 8))
sns.boxplot(x='Age', y='Heart Attack Risk', hue='Sex', data=df)
plt.title('Distribution of Heart Attack Risk by Age and Gender')
plt.xlabel('Age')
plt.ylabel('Heart Attack Risk')
plt.xticks(rotation=45)
plt.show()

#%%
# Statistical Testing: Two-Way ANOVA
# model = ols('Heart Attack Risk ~ C(Age) + C(Sex)', data=df).fit()
# anova_results = sm.stats.anova_lm(model, typ=2)
# print(anova_results)

# %%
# #checking the datafram again 
print(df)

#%%[markdown]
print(df.info()) 

#%%[markdown]
print((df).value_counts())

# %%
df.describe()
df.columns

# %%[markdown]
# # Checking normalization of the dataframe 

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# # Plotting the density of the entire DataFrame
df.plot(kind='density')
plt.title('Density Plot for the Entire DataFrame')
plt.show()

# # Plotting the density of the 'Age' column
df['Age'].plot(kind='density')
plt.title('Density Plot for the Age Column')
plt.show()

# # Plotting the density of the 'Sex' Column
df['Sex'].plot(kind='density')
plt.title("Density Plot for Sex Column")
plt.show()


# # Plotting the density  of "Cholesterol" column

df['Diabetes'].plot(kind='density')
plt.title("Density Plot for Diabetes")
plt.show()


# # Plotting the density of Alcohol Consumption
#df['Heart Attack Risk'].plot('density')
#plt.title("Heart Attack Risk")
#plt.show()

# %%[markdown]
# # Standardization and Normalization of the dataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from numpy import set_printoptions


X = df[['Age', 'Sex', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet', 'Medication Use', 'Stress Level', 'Sedentary Hours Per Day', 'Physical Activity Days Per Week', 'Sleep Hours Per Day']]
y = df['Heart Attack Risk']

scaler = MinMaxScaler(feature_range=(0, 1))
rescale = scaler.fit_transform(X)  # Apply fit_transform to the feature matrix X

set_printoptions(precision=3)

# Converting it back to DataFrame
rescaleDf = pd.DataFrame(rescale, columns=['Age', 'Sex', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet', 'Medication Use', 'Stress Level', 'Sedentary Hours Per Day', 'Physical Activity Days Per Week', 'Sleep Hours Per Day'])

# Adding back the y variable
rescaleDf['Heart Attack Risk'] = y

print(rescaleDf)

# %%[markdwon]
# # PerFormining Normalization On the DataFrame
from sklearn.preprocessing import Normalizer
from numpy import set_printoptions
import pandas as pd 

X = rescaleDf[['Age', 'Sex', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Smoking',
               'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet',
               'Medication Use', 'Stress Level', 'Sedentary Hours Per Day',
               'Physical Activity Days Per Week', 'Sleep Hours Per Day']]
y = rescaleDf['Heart Attack Risk']

scalerN = Normalizer().fit(X)
reNormalizeX = scalerN.transform(X)

set_printoptions(precision=3)
print(reNormalizeX)

# Converting it back to a DataFrame 
reNormalizeDf = pd.DataFrame(reNormalizeX, columns=['Age', 'Sex', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Smoking',
                                                   'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet',
                                                   'Medication Use', 'Stress Level', 'Sedentary Hours Per Day',
                                                   'Physical Activity Days Per Week', 'Sleep Hours Per Day'])

# Adding back y
reNormalizeDf['Heart Attack Risk'] = y

print(reNormalizeDf)

#%%[markdown]
print(reNormalizeDf.columns)
print(reNormalizeDf.info())

 

# %%[markdown]
# # Performing correlation on the standardized and normalized dataset
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming reNormalizeDf is your DataFrame after normalization

# Plotting the correlation heatmap
sns.heatmap(data=reNormalizeDf.corr(), annot=True)

# Display the plot
plt.show()

# %%[markdown]
# # Splitting and preparing the dataset  X,Y for training and testing set
X=reNormalizeDf[["Age", "Sex", 'Cholesterol', 'Heart Rate', 'Diabetes', "Smoking", "Obesity", 'Alcohol Consumption', 'Exercise Hours Per Week', "Diet", "Medication Use", "Stress Level", 'Sedentary Hours Per Day', "Physical Activity Days Per Week", "Sleep Hours Per Day"]]
print(type(X))
print(X.head(5))



y= reNormalizeDf["Heart Attack Risk"]
print(type(y))
print(y.head(5))

#%%[markdown]
# # Feature  Selection and  train-test Splitting on the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Checking shape of the X_train and y_train
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)

#%%[markdown]
# # Ensuring  y_train is 1D array 
# If y_train is a DataFrame, convert it to a 1D array
y_train = y_train.values.ravel()



#%%[markdown]
# # Performing Logistics Regression on the dataset with this features 
from sklearn.linear_model  import LogisticRegression
logitR= LogisticRegression()   #instantiating

# # Fitting my Model

logitR.fit(X_train, y_train)  ##fitting the dataset

#%%[markdown]
# # Model Evaluation (Accuracy Score)

print("Logistics Model Accuracy with test set :",  logitR.score(X_test, y_test))
print('Logistics Model Accuracy with the train set:',   logitR.score(X_train, y_train))

# # Accuracy Score explanation:
# The logistic regression model exhibits an 
# accuracy of approximately 64.18% on both the test and training sets.
# This accuracy signifies the proportion of correctly predicted outcomes
# regarding Heart Attack Risk. The consistency in accuracy between the 
# test and training sets suggests a balanced model performance. 
# if not balance its might leads to overfitting 
# However, it's essential to consider additional evaluation metrics, 
# such as precision, recall, and the confusion matrix, to gain a more
# comprehensive understanding of the model's effectiveness, particularly
# if the dataset has imbalances or specific types of errors are 
# of greater significance in the given context.



#%%[markdown]
# # Predictions
print(logitR.predict(X_train))

print("The probability of prediction rate on X_train is:", logitR.predict_proba(X_train[:15]))
print("The probability of prediction rate on X_test is:", logitR.predict_proba(X_test[:15]))

# %%[markdown]
# # Model Evaluation (Confusion Matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, logitR.predict(X_test)))

# %%[markdown]
# # Explanation 
#The classification report provides a detailed assessment of the 
# logistic regression model's performance:

#Class 0 (No Heart Attack Risk):

# Precision: 64% of instances predicted as class 0 were correct.
# Recall: All instances of actual class 0 were correctly identified.
# F1-Score: A balanced measure of precision and recall is 0.78.
# Class 1 (Heart Attack Risk):

# Precision: None of the instances predicted as class 1 were correct (precision is 0%).
# Recall: None of the actual instances of class 1 were correctly identified (recall is 0%).
# F1-Score: Due to low precision and recall, the F1-Score is 0%.
# Overall Model Performance:

# Accuracy: The model's overall accuracy on the test set is 64%.
# Warning: There is a warning about undefined metrics for class 1, indicating that the model failed to predict any instances of class 1.
# This suggests that the model performs reasonably well for class 0 but faces challenges in accurately predicting instances of class 1, potentially due to imbalances in the dataset. Addressing class imbalances and exploring adjustments to the classification threshold may be beneficial for improving performance on the minority class.

#%%[markdown]
# # Model Evaluation [compute the ROC curve and calculate AUC-ROC:]
from sklearn.metrics import roc_curve, auc

# Get predicted probabilities for class 1
y_probs = logitR.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Compute AUC-ROC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

#%%[markdown]
# # Explanation
# In summary, a ROC-AUC value of 0.50 indicates that the model's
# ability to distinguish between positive and negative classes is no
# better than random chance. The ROC curve, with a diagonal line 
# representing randomness, suggests that the model is not effectively 
# discriminating between classes at different thresholds.
# This situation may arise from the model making predictions 
# randomly or struggling to differentiate between the classes. 
# Practical implications include the need for further investigation
# into potential issues with features, model complexity, or data quality.
# Consistently low AUC values suggest that the model is not capturing 
# underlying patterns, prompting a reevaluation of feature selection,
# data preprocessing, or exploration of alternative models. Additionally,
# it is crucial to consider other evaluation metrics like precision, 
# recall, and the F1-score, especially in the context of imbalanced 
# datasets or specific dataset characteristics.

#%%[markdown]
# # Considering  Another model 
# Random Forest Classifier:

# Random Forest is an ensemble learning method that builds 
# multiple decision trees and merges them together to get a 
# more accurate and stable prediction. It often works well for both
# classification and regression tasks, handling non-linearity and
# complex relationships.

from sklearn.ensemble import RandomForestClassifier

# Instantiate the model
rf_model = RandomForestClassifier()

# Fit the model to the training data
rf_model.fit(X_train, y_train)

# Evaluate the model
print("Random Forest Model Accuracy with test set:", rf_model.score(X_test, y_test))

#%%[markdown]
# # Explanation 
# The Random Forest model achieved an accuracy of approximately 
# 62.7% on the test set, indicating that it correctly predicted the 
# heart attack risk for about two-thirds of the instances. 
# While accuracy is a standard metric, it's essential to 
# consider additional evaluation measures like precision, 
# recall, and F1-score, especially in scenarios with imbalanced datasets.
# A more in-depth analysis, including the confusion matrix, 
# can provide insights into the model's performance for each 
# class and guide improvements. Overall, the model's accuracy is moderate, but a comprehensive evaluation is necessary for a nuanced understanding of its effectiveness.


#%%[markdown]
# # Another Evaluation for random forest model (precision)
from sklearn.metrics import precision_score

# Assuming 'y_test' contains the true labels and 'predictions_rf' contains the predicted labels by the Random Forest model
predictions_rf = rf_model.predict(X_test)

# Calculate precision
precision_rf = precision_score(y_test, predictions_rf)

print(f"Precision for Random Forest: {precision_rf}")

#%%[markdown]
# # Explanation 
# A precision score of approximately 41.3% indicates a moderate
# level of accuracy in the positive predictions made by the 
# model. This means that when the model predicts a positive 
# outcome, it is correct about 41.3% of the time. 
# The precision score is one aspect of the trade-off between 
# precision and recall, and the impact depends on the specific 
# application. In situations where false positives are a concern,
# there is room for improvement in precision, and consideration 
# should be given to the overall trade-offs between different
# evaluation metrics


#%%[markdown]
# Considering another model 
# # Gradient Boosting Classifier:

# Gradient Boosting builds an ensemble of decision trees sequentially,
# where each tree corrects the errors of the previous one. It is known 
# for its high predictive accuracy.
from sklearn.ensemble import GradientBoostingClassifier

# Instantiate the model
gb_model = GradientBoostingClassifier()

# Fit the model to the training data
gb_model.fit(X_train, y_train)

# Evaluate the model
print("Gradient Boosting Model Accuracy with test set:", gb_model.score(X_test, y_test))

# # Explanation 
# The Gradient Boosting Classifier achieved a test set accuracy of 
# approximately 63.7%. This ensemble learning technique sequentially 
# builds a series of weak learners to correct errors made by previous
# models, resulting in a robust predictive model. The accuracy of 63.7% 
# implies that the model correctly predicted heart attack risk for around two-thirds of instances in the test set. To comprehensively evaluate performance, it is recommended to consider additional metrics such as precision, recall, and the F1-score. Additionally, comparing the Gradient Boosting model's performance with other models used in the analysis will help determine its relative effectiveness

#%%[markdown]
# # Performing another Evaluation for gradient Boosting Model 
from sklearn.metrics import precision_score

# Assuming 'y_test' contains the true labels and 'predictions' contains the predicted labels
predictions = gb_model.predict(X_test)

# Calculate precision
precision = precision_score(y_test, predictions)

print(f"Precision: {precision}")

#%%[markdown]
# # Explanation 
# precision score of 0.3 indicates that the model's positive 
# predictions are accurate only 30% of the time. 
# This suggests a relatively high number of false positives, 
# where instances predicted as positive are not actually true positives.
# The impact of this low precision depends on the specific application, 
# and addressing it may be crucial in scenarios where false positives 
# are costly. It's essential to consider precision in conjunction with
# other metrics and the overall context of the problem to make informed
# decisions about the model's performance



#%%[markdown]
# # Future research and suggestion 
# The future research directions for improving heart risk 
# prediction models involve a multi-faceted approach.
# Firstly, there is a need to delve deeper into feature 
# engineering, exploring new variables and transformations 
# that can better capture the intricate dynamics of heart 
# risk factors. Additionally, addressing class imbalance 
# through advanced techniques and fine-tuning model
# hyperparameters can significantly enhance predictive
# accuracy. Collaborating with domain experts, implementing 
# ensemble methods, and conducting in-depth feature importance
# analyses are crucial steps. Moreover, considering 
# interpretable models, exploring personalized prediction
# approaches, and ensuring ethical deployment underscore 
# the commitment to advancing both accuracy and transparency 
# in heart risk predictions. Continuous monitoring, 
# external validation, and a focus on ethical considerations 
# to the holistic improvement of these models for real-world 
# healthcare applications.







# %%
