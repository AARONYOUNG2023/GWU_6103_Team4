#%% [markdown]
#  ## Team 4- Final Project- Assessing the Impact of Lifestyle Factors on Heart Attack Risk in Global Populations
#
##  *Project Overview:*
#
#  Research aims to understand and combat the global challenge of heart attacks. The "Heart 
#  Attack Risk Prediction Dataset" is an excellent resource for study and analysis in healthcare and 
#  medical data science. The dataset's main goal is to predict the risk of heart attacks in individuals 
#  based on a variety of health-related characteristics.
#
##  *Dataset Introduction:*
#  
#  We collected data from Kaggle.com, and chose the dataset about heart attack prediction. The link is
#  https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset/data. 
#  
#  This set of data, which includes 8763 information from patients worldwide, comes 
#  to an end in a substantial binary classification component that indicates the existence or failure 
#  of a heart attack risk, giving a complete resource for predictive analysis and cardiovascular 
#  health research.
# 
##  *Code File Introduction:*
#  
#  This python code file includes the below parts to support our research.
#  
#  Part 1: Load and Inspect the Dataset
#
#  Part 2: Data Cleaning and preprocessing
#
#  Part 3: Exploratory Data Analysis(EDA)
#
#  Part 4: Feature Selection
#
#  Part 5: Modeling Selection and Training
#
#  Part 6: Model Evaluation (This part is included in the part 5)
#
## *Data Columns Selection:*
#
#  We selected some columns from the original dataset, because there are some unrelated columns for
#  our topic. We deleted the all unrelated columns before we imported data into python.
#
#%% [markdown]
# ## Loading the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import os

# %% [markdown]
# ## Part 1: Load and Inspect the Dataset
# Importing dataset from CSV and show the dataset
df = pd.read_csv('heart_attack_prediction_dataset_revised.csv', delimiter=';', on_bad_lines = 'skip')
df

# %%
# Displaying the first few rows of the dataset
print(df.head())

#%%
# Displaying the info of the dataset
print(df.info())

#%%
# Summary statistics of numerical features
print(df.describe())

#%% [markdown]
# ## DataFrame Information Understanding
#
# From checking the basic dataset information, we found that as for preprocessing and cleaning, 
# several columns may require attention. For instance, 'Age', 'Cholesterol', 'Heart Rate', 
# 'Stress Level', 'Sedentary Hours Per Day', 'Physical Activity Days Per Week', and 'Sleep Hours 
# Per Day' are numerical and may need to be normalized or standardized to ensure consistent scale, 
# Additionally, the 'Diet' column, being a categorical variable with multiple categories, might 
# require encoding (like one-hot encoding) to convert it into a numerical format suitable for analysis. 
#
# |Column Name  |Description	|Data Type	|Categories (If Categorical)
# 
# |Patient ID	|Unique identifier for patient	|Categorical	|Unique IDs
#
# |Age	        |Age of the patient	|Numerical	|N/A
#
# |Sex	        |Gender of the patient	|Categorical	|Male, Female
#
# |Cholesterol	|Cholesterol level	|Numerical	|N/A
#
# |Heart Rate	|Patient's heart rate	|Numerical	|N/A
#
# |Diabetes	    |Diabetes status	|Categorical	|0: No, 1: Yes
#
# |Smoking	    |Smoking status	    |Categorical	|0: No, 1: Yes
#
# |Obesity	    |Obesity status	    |Categorical	|0: No, 1: Yes
#
# |Alcohol Consumption	|Alcohol use	|Categorical	|0: No, 1: Yes
#
# |Exercise Hours Per Week	|Weekly exercise hours	|Numerical	|N/A
#
# |Diet	        |Type of diet	    |Categorical	|ex: Healthy, Unhealthy, Average
#
# |Medication Use	|Medication usage	|Categorical	|0: No, 1: Yes
# 
# |Stress Level	|Level of stress	|Numerical	|N/A
#
# |Sedentary Hours Per Day	|Daily sedentary hours	|Numerical	|N/A
#
# |Physical Activity Days Per Week	|Active days per week	|Numerical	|N/A
#
# |Sleep Hours Per Day	|Daily sleep hours	|Numerical	|N/A
#
# |Heart Attack Risk	|Risk of heart attack	|Categorical	|0: Low, 1: High
#

#%% [markdown]
# ## Part 2: Data Cleaning and preprocessing
# *2.1 - Data Cleaning*
#
# *2.1.1 - Checking the missing values*
#
print("Missing Values:")
print(df.isnull().sum())

#%%
# *2.1.2 - Droping rows with missing values if it has*
df = df.dropna()

# *2.1.3 - Droping the column 'Patient ID'
df = df = df.drop(columns=['Patient ID'])

# *2.1.4 - Displaying cleaned dataset*
print("Cleaned Dataset:\n")
print(df.head())

#%%
# *2.1.5 - Checking the columns name again*
print("Column Names:", df.columns)

# %% [markdown]
# *2.2 - Data preprocessing*
#
# *2.2.1 - Converting Categorical Data*
# 
# In the dataset, the columns 'sex' and 'Diet' have categorical data, so we need to convert them
# into numerical data to ensure we can use them in the following modeling part
#
# Mapping 'Male' to 1 and 'Female' to 0 in the 'Sex' column
#
sex_mapping = {'Male': 1, 'Female': 0}
df['Sex'] = df['Sex'].map(sex_mapping)

# Map 'Unhealthy' to 0, 'Average' to 1, and 'Healthy' to 2 in the 'Diet' column
# 
diet_mapping = {'Unhealthy': 0, 'Average': 1, 'Healthy': 2}
df['Diet'] = df['Diet'].map(diet_mapping)

#%%
# After converting those data into numerical data, checking the df again
#
df

#
# It looks like all be numerical data, so we can do the further research.

# %% [markdown]
# ## Part 3: Exploratory Data Analysis(EDA)
#
# Checking the summary statistics again
#
print(df.describe())
#%%
# *3.1 - Histograms for Numeric Variables*
#
# List of numerical variables
#
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

#%% [markdown]
# *3.1 Histogram of Numerical Data Interpretation*
#
#  1. Age: Uniformly spread, slight periodic peaks.
#
#  2. Cholesterol: Roughly normal, right-skewed indicating higher values.
#
#  3. Heart Rate: Near normal, few high values.
#
#  4. Exercise Hours: Left-skewed, most exercise little.
#
#  5. Sedentary Hours: Broad spread, regular peaks.
#
#  6. Physical Activity Days: Peaks at 0, 3, and 5-7 days suggest varied activity levels.
#
#  7. Sleep Hours: Bimodal, peaks at 6 and 8 hours, common sleep durations.
#
# %% [markdown]
# *3.2 - Bar Charts for Categorical Variables*
#
# Exclude the numerical variables and identifiers to get the categorical ones
excluded_vars = numerical_vars + ['Patient ID']  
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

#%% [markdown]
# *3.2 - Bar Charts of categorical Data Interpretation*
# 
# 1. Alcohol Consumption: More individuals do not consume alcohol compared to those who do.
#
# 2. Diabetes: A larger number of individuals do not have diabetes.
#
# 3. Diet: The distribution is even across diet categories, suggesting a balance between different diet types.
#
# 4. Heart Attack Risk: Fewer individuals are at risk of heart attack compared to those not at risk.
#
# 5. Medication Use: Medication use is less common than non-use.
#
# 6. Obesity: More individuals are not obese than those who are.
#
# 7. Sex: The distribution between genders is roughly even.
#
# 8. Smoking: Fewer individuals smoke compared to those who do not smoke.
#
# 9. Stress Level: Stress levels are evenly distributed across the scale from 1 to 10. 
#
#%% [markdown]
# *3.3 - Correlation Matrix for numerical data
#
# *3.3.1 - Correlation Matrix Graph
#
# This code used select_dtypes to include only numeric columns in the correlation matrix computation.
#
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
matrix_correlation = df_numeric.corr()

# Use Seaborn's heatmap to plot the correlation matrix.
plt.figure(figsize=(10, 8))
sns.heatmap(matrix_correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

#%% 
# *3.3.2 - Highest Correclation Coefficients
#
# Finding the features that are strongest predictors of heart attack risk, in decreasing order.
#
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
print("Top 5 Correlated Features with Heart Attack Risk:\n")
print(top5_features)

# %%
# Filter rows where "Heart Attack Risk" is 1
heart_attack_df = df[df['Heart Attack Risk'] == 1]

top5_features = sorted_correlation.index[1:6]  # Exclude 'Heart Attack Risk' itself

# Plot histograms or bar plots for the top 5 correlated features
for feature in top5_features:
    if df[feature].dtype == 'object' or df[feature].nunique() <= 10:  # Categorical data
        plt.figure(figsize=(8, 6))
        sns.countplot(x=feature, data=heart_attack_df, color='red')
        plt.title(f'Distribution of {feature} for Heart Attack Risk = 1')
        plt.xlabel(feature)
        plt.ylabel('Count')
    else:  # Continuous data
        plt.figure(figsize=(8, 6))
        sns.histplot(heart_attack_df[feature], bins=20, kde=True, color='red')
        plt.title(f'Histogram of {feature} for Heart Attack Risk = 1')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
    plt.show()

#%% [markdown]
# *3.3.2 - Plot Histograms or bar plots Interpretation
#
# 1. Cholesterol Histogram: The distribution of cholesterol among high-risk individuals 
# shows variability with a range of peaks, indicating no single dominant cholesterol level 
# associated with increased heart attack risk.
#
# 2. Sleep Hours Histogram: Sleep duration for high-risk individuals is fairly evenly 
# distributed from 4 to 9 hours, with no specific sleep duration appearing to be 
# significantly more common in this group.
#
# 3. Diabetes Histogram: A greater number of individuals at high risk for heart 
# attacks are diabetic, with diabetic individuals outnumbering non-diabetics almost 
# 2 to 1.
#
# 4. Alcohol Consumption Histogram: More individuals at high risk for heart attacks 
# consume alcohol than do not, suggesting a potential link between alcohol consumption 
# and increased heart attack risk.
#
# 5. Obesity Histogram: The distribution between obese and non-obese individuals 
# in the high-risk category is nearly even, suggesting obesity is a common trait 
# among those at high risk for heart attacks.
#

#%% [markdown]
# *3.4 - Checking the specific feature with Heart Attack Risk
#
# *3.4.1 - The relation between Cholesterol and Heart Attack Risk
#
# After checking the above EDA, we would like to check whether 'Cholesterol' is related to
# the heart risk attack
#
plt.figure(figsize=(14, 8))
# Creating a subplot for the regplot
plt.subplot(1, 2, 1)
sns.regplot(x='Cholesterol', y='Heart Attack Risk', data=df, logistic=True, scatter_kws={'s': 20}, color='pink')
plt.title('Relationship Between Cholesterol Levels and Heart Attack Risk')
plt.xlabel('Cholesterol Levels')
plt.ylabel('Heart Attack Risk')

# Creating a subplot for the boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x='Heart Attack Risk', y='Cholesterol', data=df, color='purple')
plt.title('Box Plot of Cholesterol Levels by Heart Attack Risk')
plt.xlabel('Heart Attack Risk')
plt.ylabel('Cholesterol Levels')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
#%% [markdown]
# *3.4.1 - The regplot and boxplot between Cholesterol and Heart Attack Risk Interpretation
#
# In the 1st regplot, it shows Cholesterol has a positive affect on Heart Attack Risk. In the
# 2nd boxplot, the Heart Attack Risk =1 looks like to have a higher Cholesterol looks.
#
#%% [markdown]
# *3.4.2 - The relation between age or sex and Heart Attack Risk*
#
# Define age groups
age_bins = [0, 29, 39, 49, 59, 69, 79, 89, 99]
age_labels = ['0-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
#
# Create a subplot layout
fig, axes = plt.subplots(3, 1, figsize=(15, 18))

# Create a line plot for the median heart attack risk by age group and gender
sns.lineplot(data=df, x='Age Group', y='Heart Attack Risk', hue='Sex', estimator='median', ax=axes[0])
axes[0].set_title('Line Plot of Median Heart Attack Risk by Age Group and Gender')
axes[0].set_xlabel('Age Group')
axes[0].set_ylabel('Median Heart Attack Risk')

# Create a violin plot for heart attack risk by age group and gender
sns.violinplot(data=df, x='Age Group', y='Heart Attack Risk', hue='Sex', split=True, inner="quart", ax=axes[1])
axes[1].set_title('Violin Plot of Heart Attack Risk by Age Group and Gender')
axes[1].set_xlabel('Age Group')
axes[1].set_ylabel('Heart Attack Risk')

# Create a box plot for heart attack risk by age group and gender
sns.boxplot(data=df, x='Age Group', y='Heart Attack Risk', hue='Sex', ax=axes[2])
axes[2].set_title('Box Plot of Heart Attack Risk by Age Group and Gender')
axes[2].set_xlabel('Age Group')
axes[2].set_ylabel('Heart Attack Risk')

# Adjust the layout
plt.tight_layout()
plt.show()
# %% [markdown]
# *3.4.2 - The plots between age or sex and Heart Attack Risk Interpretation*
#
# From the above plots, we can see that sex and age cannot be the strong individual
# predictors of heart attack risk.
#
#%% [markdown]
# *3.4.3 - Analyzing Diabetes Prevalence in Obese vs. Non-Obese by Age*
#
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
from scipy.stats import chi2_contingency
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

# %% [markdown]
# *3.4.3 - Interpretation*
#
# In all age groups, the p-values are greater than 0.05, indicating that there is no 
# statistically significant association between obesity and diabetes within these
# specific age categories. However, the '60-69' age group shows a p-value closest 
# to 0.05, suggesting a potential trend that might be worth exploring further, 
# especially in a larger dataset or a more focused study.
#
#%% [markdown]
# ## Part 4: Features Selection
#
# From the part 3: EDA, we can see that there is no an individual factor with a strong
# relationship with Heart Attack Risk, so we decide to use all features to build a model
# to check the accuracy, precision, or other factors.
#
#%% [markdwon]
# ## Part 5: Modeling preparation, selection, and training
#
# *5.1 - Modeling Preparation(Data Normalization and Scaling)*
#
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


X = df[['Age', 'Sex', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Smoking', 'Obesity', 
        'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet', 'Medication Use', 
        'Stress Level', 'Sedentary Hours Per Day', 'Physical Activity Days Per Week', 
        'Sleep Hours Per Day']]
y = df['Heart Attack Risk']

scaler = MinMaxScaler(feature_range=(0, 1))
rescale = scaler.fit_transform(X)  # Apply fit_transform to the feature matrix X

set_printoptions(precision=3)

# Converting it back to DataFrame
rescaleDf = pd.DataFrame(rescale, columns=['Age', 'Sex', 'Cholesterol', 'Heart Rate', 
                                           'Diabetes', 'Smoking', 'Obesity', 
                                           'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet', 
                                           'Medication Use', 'Stress Level', 'Sedentary Hours Per Day', 
                                           'Physical Activity Days Per Week', 'Sleep Hours Per Day'])

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

#%%
# *5.2 - Data Splitting*
#
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
X=reNormalizeDf[["Age", "Sex", 'Cholesterol', 'Heart Rate', 'Diabetes', "Smoking", 
                 "Obesity", 'Alcohol Consumption', 'Exercise Hours Per Week', "Diet", 
                 "Medication Use", "Stress Level", 'Sedentary Hours Per Day', 
                 "Physical Activity Days Per Week", "Sleep Hours Per Day"]]
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

#%%
# *5.3 - Modeling*
#
# *5.3.1 - Logistic Model*
#
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

#%%
# *5.3.2 - Random Tree Model
#
# Random Forest Classifier:
#
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

#%% [markdown]
# *5.3.3 - Gradient Boosting Model*
#
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
# ## Explanation 
# The precision score of 0.3 indicates that the model's positive 
# predictions are accurate only 30% of the time.
#
# This suggests a relatively high number of false positives, 
# where instances predicted as positive are not actually true positives.
# The impact of this low precision depends on the specific application, 
# and addressing it may be crucial in scenarios where false positives 
# are costly. It's essential to consider precision in conjunction with
# other metrics and the overall context of the problem to make informed
# decisions about the model's performance


#%% [markdown]
# ## Future research and suggestion 
# The future research directions for improving heart risk 
# prediction models involve a multi-faceted approach.
#
# Firstly, there is a need to delve deeper into feature 
# engineering, exploring new variables and transformations 
# that can better capture the intricate dynamics of heart 
# risk factors.
#
# Additionally, addressing class imbalance 
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

#%%




