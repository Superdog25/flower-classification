
# Flower Classification Project (Iris Dataset)

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# ------------------ Step 1: Load and Explore Dataset ------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("First 5 Rows of Dataset:")
print(df.head())
print("\nDataset Summary:")
print(df.describe())
print("\nClass Distribution:")
print(df['species'].value_counts())

# ------------------ Step 2: Visualization ------------------
sns.pairplot(df, hue='species', diag_kind='kde')
plt.suptitle("Iris Feature Pair Relationships", y=1.02)
plt.savefig('pairplot.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.heatmap(df.drop(columns='species').corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlations")
plt.savefig('correlation_heatmap.png')
plt.close()

# ------------------ Step 3: Train Models ------------------
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

lr = LogisticRegression(max_iter=200)
dt = DecisionTreeClassifier(random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)

print("\nModel Accuracies:")
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

# ------------------ Step 4: Confusion Matrices ------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test, ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title("Logistic Regression Confusion Matrix")

ConfusionMatrixDisplay.from_estimator(dt, X_test, y_test, ax=axes[1], cmap='Greens', colorbar=False)
axes[1].set_title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

print("\nClassification Report (Decision Tree):")
print(classification_report(y_test, dt_pred))

# ------------------ Step 5: CLI Prediction ------------------
def predict_species(features):
    pred = dt.predict([features])
    return iris.target_names[pred][0]

if __name__ == "__main__":
    if len(sys.argv) == 5:
        features = list(map(float, sys.argv[1:]))
        species = predict_species(features)
        print(f"\n🌸 Predicted Iris Species: {species.capitalize()}")
    else:
        print("\nTo predict a new flower species, use:")
        print("python flower_classification.py <sepal_length> <sepal_width> <petal_length> <petal_width>")
        print("Example: python flower_classification.py 5.1 3.5 1.4 0.2")
