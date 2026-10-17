"""
Loads the raw dataset from the repo's data folder, applies the
required cleaning/feature steps (including missing-value handling),
and splits it into train/test sets. Runs as the data-prep job of the
pipeline; the resulting CSVs are handed to the next job as a workflow
artifact.
"""
import os
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "superkart_project/data/SuperKart.csv"
TARGET_COLUMN = "Product_Store_Sales_Total"

PERISHABLE_ITEMS = [
    "Dairy",
    "Meat",
    "Seafood",
    "Fruits and Vegetables",
    "Frozen Foods",
    "Breads",
]


def categorize_food(product_type):
    return "Perishable" if product_type in PERISHABLE_ITEMS else "Non Perishable"


def prepare_data():
    df = pd.read_csv(RAW_PATH)

    # Standardise sugar-content labels
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace("reg", "Regular")

    # Store age from establishment year
    current_year = datetime.now().year
    df["Store_Age"] = current_year - df["Store_Establishment_Year"]

    # Perishable vs non-perishable flag
    df["type_of_food"] = df["Product_Type"].apply(categorize_food)

    # Drop identifier / now-redundant columns
    df = df.drop(columns=["Product_Id", "Product_Type", "Store_Establishment_Year"])

    # ---- Missing value check + handling ----
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    print("Missing values before cleaning:")
    print(missing_counts if not missing_counts.empty else "None")

    if not missing_counts.empty:
        # Rows missing the target can't be used for training/evaluation.
        before_rows = len(df)
        df = df.dropna(subset=[TARGET_COLUMN])
        dropped = before_rows - len(df)
        if dropped:
            print(f"Dropped {dropped} row(s) missing the target column.")

        # For remaining missing values: numeric -> median, categorical -> mode.
        numeric_cols = df.select_dtypes(include="number").columns
        categorical_cols = df.select_dtypes(exclude="number").columns

        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled missing '{col}' with median ({median_val}).")

        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode().iloc[0]
                df[col] = df[col].fillna(mode_val)
                print(f"Filled missing '{col}' with mode ('{mode_val}').")

    print("\nMissing values after cleaning:", int(df.isnull().sum().sum()))
    # ---- End missing value handling ----

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("\nData preparation complete.")
    print(f"Xtrain shape: {Xtrain.shape}   Xtest shape: {Xtest.shape}")
    print("\nColumns after cleaning:", list(X.columns))
    print("\ntype_of_food distribution (train):")
    print(Xtrain["type_of_food"].value_counts())


if __name__ == "__main__":
    prepare_data()
