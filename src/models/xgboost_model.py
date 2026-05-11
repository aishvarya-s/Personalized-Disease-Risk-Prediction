import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

# ---------------- LOAD DATASET ---------------- #

print("Loading dataset...")

df = pd.read_csv("data/processed/sepsis_dataset.csv")

# ---------------- FEATURE ENGINEERING ---------------- #

df["shock_index"] = (
    df["heart_rate"] /
    (df["systolic_bp"] + 1)
)

df["pulse_pressure"] = (
    df["systolic_bp"] -
    df["diastolic_bp"]
)

# ---------------- FEATURES ---------------- #

feature_cols = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
    "shock_index",
    "pulse_pressure",
    "respiratory_rate",
    "spo2",
    "gender",
    "anchor_age"
]

X = df[feature_cols]
y = df["sepsis"]

# ---------------- CLEAN DATA ---------------- #

X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(X.mean())

# ---------------- TRAIN TEST SPLIT ---------------- #

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- FEATURE SCALING ---------------- #

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- XGBOOST MODEL ---------------- #

print("Training XGBoost model...")

model = XGBClassifier( n_estimators=400,
                       max_depth=10, 
                       learning_rate=0.03, 
                       subsample=0.85, 
                       colsample_bytree=0.85, 
                       scale_pos_weight=2.5, 
                       min_child_weight=3, 
                       gamma=0.1, 
                       random_state=42, 
                       eval_metric="logloss" )

model.fit(X_train, y_train)

# ---------------- PREDICTIONS ---------------- #

print("Making predictions...")

y_pred = model.predict(X_test)

# ---------------- EVALUATION ---------------- #

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------- FEATURE IMPORTANCE ---------------- #

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)