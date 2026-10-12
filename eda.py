
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Churn_Modelling.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# Create the output folder if it doesn't exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load the dataset
df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nChurn distribution:")
print(df["Exited"].value_counts())


# 1. Overall churn distribution
fig, ax = plt.subplots(figsize=(7, 5))

churn_labels = df["Exited"].map({0: "Stayed", 1: "Churned"})

sns.countplot(
    x=churn_labels,
    order=["Stayed", "Churned"],
    ax=ax
)

ax.set_title("Bank Customer Churn Distribution")
ax.set_xlabel("Customer Status")
ax.set_ylabel("Number of Customers")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "churn_distribution.png", dpi=150)
plt.close(fig)


# 2. Churn rate by age group
age_bins = [0, 25, 35, 45, 55, 65, 120]
age_labels = ["25 or younger", "26–35", "36–45", "46–55", "56–65", "66+"]

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=age_bins,
    labels=age_labels
)

age_churn = (
    df.groupby("AgeGroup", observed=True)["Exited"]
    .mean()
    .mul(100)
)

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    x=age_churn.index.astype(str),
    y=age_churn.values,
    color="#176B87",
    ax=ax
)

ax.set_title("Customer Churn Rate by Age Group")
ax.set_xlabel("Age Group")
ax.set_ylabel("Churn Rate (%)")
ax.tick_params(axis="x", rotation=15)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "churn_by_age_group.png", dpi=150)
plt.close(fig)


# 3. Churn rate by geography
geography_churn = (
    df.groupby("Geography")["Exited"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(7, 5))

sns.barplot(
    x=geography_churn.index,
    y=geography_churn.values,
    color="#176B87",
    ax=ax
)

ax.set_title("Customer Churn Rate by Geography")
ax.set_xlabel("Geography")
ax.set_ylabel("Churn Rate (%)")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "churn_by_geography.png", dpi=150)
plt.close(fig)


print("\nEDA completed successfully!")
print("Charts saved in:", FIGURES_DIR)
print("- churn_distribution.png")
print("- churn_by_age_group.png")
print("- churn_by_geography.png")
