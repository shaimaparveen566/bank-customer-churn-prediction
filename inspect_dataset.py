
from pathlib import Path

import pandas as pd

# Locate the dataset relative to this Python file
DATA_PATH = Path(__file__).parent / "data" / "Churn_Modelling.csv"

# Load the CSV file
df = pd.read_csv(DATA_PATH)

# 1. Show the first five rows
print("\n--- FIRST FIVE ROWS ---")
print(df.head())

# 2. Show the number of rows and columns
print("\n--- DATASET SHAPE ---")
print(df.shape)

# 3. Show column names and data types
print("\n--- DATASET INFORMATION ---")
df.info()

# 4. Check for missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 5. Check for duplicate rows
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

# 6. Check the target column distribution

# Count customers who stayed and customers who left
print("\n--- CHURN COUNTS ---")
print(df["Exited"].value_counts().sort_index())

# Show percentages
print("\n--- CHURN PERCENTAGES ---")
print((df["Exited"].value_counts(normalize=True).sort_index() * 100).round(2))


