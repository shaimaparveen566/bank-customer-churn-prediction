
from pathlib import Path

import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# 1. Load the dataset
DATA_PATH = (
    Path(__file__).parent
    / "data"
    / "Churn_Modelling.csv"
)
df = pd.read_csv(DATA_PATH)

# 2. Remove columns that should not be model inputs
X = df.drop(
    columns=["RowNumber", "CustomerId", "Surname", "Exited"]
)
y = df["Exited"]

# 3. Identify numerical and categorical features
categorical_features = ["Geography", "Gender"]
numerical_features = [
    column for column in X.columns
    if column not in categorical_features
]

# 4. Prepare each type of feature
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("numerical", StandardScaler(), numerical_features),
    ]
)

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 6. Build the machine learning pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

# 7. Train the model
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate performance
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
report = classification_report(
    y_test,
    y_pred,
    zero_division=0,
)

print("\nClassification Report:")
print(report)

# Save the classification report
OUTPUT_PATH = Path(__file__).parent / "classification_report.txt"

with open(OUTPUT_PATH, "w") as file:
    file.write("Bank Customer Churn Prediction\n")
    file.write("Model: Random Forest\n\n")
    file.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n\n")
    file.write("Confusion Matrix:\n")
    file.write(str(confusion_matrix(y_test, y_pred)))
    file.write("\n\nClassification Report:\n")
    file.write(report)

print(f"\nResults saved to: {OUTPUT_PATH}")

# Save the trained model pipeline
MODEL_PATH = Path(__file__).parent / "churn_model.joblib"

joblib.dump(model, MODEL_PATH)

print(f"Trained model saved to: {MODEL_PATH}")