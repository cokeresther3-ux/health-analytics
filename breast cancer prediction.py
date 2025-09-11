import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_curve, auc

df = pd.read_csv("/Users/esthercoker/Downloads/The_Cancer_data_1500_V2.csv")

print("\nDataset shape:", df.shape)
print(df.describe())
print("\nColumns in dataset:", df.columns.tolist())

features = ["Age", "Gender", "BMI", "Smoking", "GeneticRisk",
            "PhysicalActivity", "AlcoholIntake", "CancerHistory"]
target = "Diagnosis"

X = df[features].copy()
y = df[target].copy()

#Convert to numerical 
for col in ["Gender", "GeneticRisk", "AlcoholIntake"]:
    if X[col].dtype == "object":
        X[col] = X[col].astype("category").cat.codes

print("\nData types after encoding")
print(X.dtypes)

# Correlation heatmap
df_corr = X.copy()
df_corr["Diagnosis"] = y
corr = df_corr.corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
            xticklabels=corr.columns, yticklabels=corr.columns, cbar=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Age by diagnosis
plt.figure(figsize=(6, 4))
sns.barplot(x=y, y=X["Age"], palette="Set2", ci=None)
plt.xticks([0, 1], ["No Cancer", "Cancer"])
plt.xlabel("Diagnosis")
plt.ylabel("Average Age")
plt.title("Average Age by Diagnosis")
plt.show()

# Smoking status
df["SmokingStatus"] = df["Smoking"].map({0: "Non-Smoker", 1: "Smoker"})

plt.figure(figsize=(7, 5))
sns.countplot(x="SmokingStatus", hue=y, data=df, palette="pastel")
plt.xlabel("Smoking Status")
plt.ylabel("Count")
plt.title("Smoking Status by Diagnosis")
plt.legend(title="Diagnosis", labels=["No Cancer", "Cancer"])
plt.show()

# Model Training 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(max_depth=5, random_state=42) # decision tree
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred) # accuracy score
confusion_m = confusion_matrix(y_test, y_pred) #confusion matrix

print("\nMODEL EVALUATION")

print(f"Accuracy score: {acc:.4f}")

print("\nConfusion Matrix:")
print(confusion_m)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Confusion matrix heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_m, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Cancer", "Cancer"],
            yticklabels=["No Cancer", "Cancer"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()

# Decision tree 
plt.figure(figsize=(14, 7))
plot_tree(clf, feature_names=features, class_names=["No Cancer", "Cancer"], filled=True)
plt.title("Decision Tree (max_depth=5)")
plt.tight_layout()
plt.show()

# Feature Importances
importances = clf.feature_importances_
order = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 5))
sns.barplot(x=importances[order], y=np.array(features)[order], palette="viridis")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importances (Decision Tree)")
plt.tight_layout()
plt.show()

# Predicted probabilities 
y_prob = clf.predict_proba(X_test)[:, 1]

# ROC  curve 
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="blue", lw=2, label="ROC curve (AUC = {:.2f})".format(roc_auc))
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")  
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()