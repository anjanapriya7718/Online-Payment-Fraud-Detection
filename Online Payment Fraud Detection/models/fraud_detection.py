# Online Payment Fraud Detection using ML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

# -----------------------------
# 1. Read Dataset
# -----------------------------
df = pd.read_csv("../Data/PS_20174392719_1491204439457_log.csv", nrows=200000)
print("Dataset Loaded Successfully")
print(df.head())

# -----------------------------
# 2. Drop unwanted columns
# -----------------------------
df.drop(["nameOrig","nameDest"], axis=1, inplace=True)

# -----------------------------
# 3. Label Encoding (type column)
# -----------------------------
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# -----------------------------
# 4. Define features & target
# -----------------------------
X = df.drop(["isFraud"], axis=1)
y = df["isFraud"]

# -----------------------------
# 5. Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Started...")

# -----------------------------
# 6. Models
# -----------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=50),
    "Decision Tree": DecisionTreeClassifier(),
    "Extra Trees": ExtraTreesClassifier(n_estimators=50),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

accuracy_results = {}

# -----------------------------
# 7. Train & Test Models
# -----------------------------
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracy_results[name] = acc
    print(f"{name} Accuracy:", acc)

# -----------------------------
# 8. Best Model
# -----------------------------
best_model_name = max(accuracy_results, key=accuracy_results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

# -----------------------------
# 9. Save Model
# -----------------------------
joblib.dump(best_model, "fraud_model.pkl")
print("Model saved as fraud_model.pkl")

# -----------------------------
# 10. Final Accuracy Report
# -----------------------------
pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, pred))

