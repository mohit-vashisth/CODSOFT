import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load Raw Dataset
df = pd.read_csv("../dataset/titanic_original.csv")
df.columns = df.columns.str.lower()

# Step 2: Feature Engineering - Title Extraction
df["title"] = df["name"].str.extract(" ([A-Za-z]+)\.", expand=False)
df["title"] = df["title"].replace(
    [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ],
    "Rare",
)
df["title"] = df["title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

# Step 3: Cabin Deck Extraction
df["cabin_deck"] = df["cabin"].str[0]
df["cabin_deck"] = df["cabin_deck"].fillna("U")

# Step 4: Fill Missing Embarked
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Step 5: Family Size
df["familysize"] = df["sibsp"] + df["parch"] + 1

# Step 6: Label Encoding
for col in ["sex", "embarked", "title"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Step 7: Name Length
df["name_length"] = df["name"].apply(len)

# Step 8: Drop Unnecessary Columns
df.drop(["name", "ticket", "passengerid", "cabin"], axis=1, inplace=True)

# Step 9: Age Imputation by Title Median
df["age"] = df.groupby("title")["age"].transform(lambda x: x.fillna(x.median()))

# Step 10: Rename for Consistency
df.rename(columns={"familysize": "family_size"}, inplace=True)

# Step 11: Save EDA-processed CSV
df.to_csv("../dataset/titanic_eda.csv", index=False)


# --------------- PHASE 2: POST-EDA PROCESSING --------------- #

df = pd.read_csv("../dataset/titanic_eda.csv")

# Step 12: Additional Features
df["is_alone"] = (df["family_size"] == 1).astype(int)
df["fare_per_person"] = df["fare"] / df["family_size"]
df["age_bin"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, 80],
    labels=["child", "teen", "young", "adult", "senior"],
)

df["class_fare"] = df["pclass"] * df["fare"]
df["fare_log"] = np.log1p(df["fare"])  # log(1 + fare)

# Step 13: One-hot Encoding Age Bin
df = pd.get_dummies(df, columns=["age_bin"], prefix="age")

# Step 14: Child-Female Indicator
df["is_child_female"] = ((df["sex"] == 0) & (df["age"] <= 12)).astype(int)

# Step 15: Final Save
df.to_csv("../dataset/titanic_processed.csv", index=False)
