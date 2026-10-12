
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset
DATA_PATH = Path(__file__).parent / "data" / "Churn_Modelling.csv"
df = pd.read_csv(DATA_PATH)

# 2. Separate input features and target
X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
y = df["Exited"]

# 3. Identify categorical and numerical columns
categorical_features = ["Geography", "Gender"]
numerical_features = [
    column for column in X.columns
    if column not in categorical_features
]

# 4. Prepare preprocessing
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

# 5. Split data consistently for both models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 6. Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=0.1,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

# 7. Train and evaluate each model
results = []

for name, classifier in models.items():
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Churn Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Churn Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Churn F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
    })

# 8. Display and save comparison
results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(results_df.round(3).to_string(index=False))

OUTPUT_PATH = Path(__file__).parent / "model_comparison.csv"
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nComparison saved to: {OUTPUT_PATH}")
