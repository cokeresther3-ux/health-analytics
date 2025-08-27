
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("/Users/esthercoker/Downloads/insurance.csv")

print("Dataset shape:", df.shape) #size of dataset 
print(df.info()) #cheking for missing data

#Description of statistics
print("\nSummary statistics:")
print(df.describe())

# one-hot encoding categorical variables to numerical
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df['gender'] = df['sex'].map({'male': 1, 'female': 0}) 
df['region'] = df['region'].map({
    'northeast': 0,
    'northwest': 1,
    'southeast': 2,
    'southwest': 3})

# Selecting all numeric columns 
numeric_df = df.select_dtypes(include=np.number)

# correlation matrix
corr_matrix = numeric_df.corr()

# Plotting heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap ")
plt.show()


# Distribution of gender
plt.figure(figsize=(6,4))
sns.countplot(x='sex', data=df)
plt.title("Distribution of Gender")
plt.show()

# Distribution  of smoker
plt.figure(figsize=(6,4))
sns.countplot(x='smoker', data=df)
plt.title("Distribution of Smokers")
plt.show()

# Distribution of age
plt.figure(figsize=(8,4))
sns.histplot(df['age'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Age")
plt.show()

# Distribution of charges 
plt.figure(figsize=(8,4))
sns.histplot(df['charges'], bins=30, kde=True, color='salmon')
plt.title("Distribution of Medical Charges")
plt.show()


# Distribution of charges by Gender
plt.figure(figsize=(6,4))
sns.boxplot(x='sex', y='charges', data=df)
plt.title("Medical Charges by Gender")
plt.show()


# Age vs Charges
plt.figure(figsize=(6,4))
sns.scatterplot(x='age', y='charges', data=df)
sns.regplot(x='age', y='charges', data=df, scatter=False, color='red')
plt.title("Age vs Medical Charges")
plt.show()

# Gender vs charges
plt.figure(figsize=(6,4))
sns.boxplot(x='sex', y='charges', data=df)
plt.title("Gender vs Medical Charges")
plt.show()

# Smoker vs charges
plt.figure(figsize=(6,4))
sns.boxplot(x='smoker', y='charges', data=df)
plt.title("Smoker vs Medical Charges")
plt.show()

# Regression line of Age vs Charges
plt.figure(figsize=(6,4))
sns.scatterplot(x='age', y='charges', data=df)
sns.regplot(x='age', y='charges', data=df, scatter=False, color='red')
plt.title("Age vs Medical Charges")
plt.xlabel("Age")
plt.ylabel("Annual medical insurance costs in US dollars")
plt.show()

#Convert categorical variables to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

#Split into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

#Prediction
y_pred = lr_model.predict(X_test)

#Evaluate Model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

#Testing accuracy of model
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

#Plot Actual vs Predicted
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Medical Charges")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.show()
