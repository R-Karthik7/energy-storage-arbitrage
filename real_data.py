import pandas as pd
import numpy as np


# ==============================
# LOAD REAL ELECTRICITY DATA
# ==============================

DATA_FILE = "energy_dataset.csv"

print("=" * 60)
print("LOADING REAL ELECTRICITY DATASET")
print("=" * 60)

# Load dataset
df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# ==============================
# CHECK REQUIRED COLUMNS
# ==============================

required_columns = ["time", "price day ahead"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' was not found in the dataset."
        )

print("\nRequired columns found:")
print("- time")
print("- price day ahead")


# ==============================
# CONVERT TIME COLUMN
# ==============================

df["time"] = pd.to_datetime(df["time"], utc=True)

# Sort chronologically
df = df.sort_values("time").reset_index(drop=True)


# ==============================
# SELECT DAY-AHEAD PRICE
# ==============================

prices = df[["time", "price day ahead"]].copy()

# Remove missing prices
prices = prices.dropna(subset=["price day ahead"])

# Remove invalid values
prices = prices[
    np.isfinite(prices["price day ahead"])
].reset_index(drop=True)


# ==============================
# DISPLAY INFORMATION
# ==============================

print("\n" + "=" * 60)
print("REAL PRICE DATA")
print("=" * 60)

print("\nNumber of valid price records:", len(prices))

print(
    "Start time:",
    prices["time"].iloc[0]
)

print(
    "End time:",
    prices["time"].iloc[-1]
)

print(
    "\nMinimum price:",
    round(prices["price day ahead"].min(), 2)
)

print(
    "Maximum price:",
    round(prices["price day ahead"].max(), 2)
)

print(
    "Average price:",
    round(prices["price day ahead"].mean(), 2)
)

print(
    "Median price:",
    round(prices["price day ahead"].median(), 2)
)


# ==============================
# CHECK FOR DUPLICATES
# ==============================

duplicate_times = prices["time"].duplicated().sum()

print("\nDuplicate timestamps:", duplicate_times)


# ==============================
# CHECK FOR MISSING VALUES
# ==============================

missing_prices = prices["price day ahead"].isna().sum()

print("Missing prices:", missing_prices)


# ==============================
# TRAIN / TEST SPLIT
# ==============================

split_index = int(len(prices) * 0.80)

train_data = prices.iloc[:split_index].copy()
test_data = prices.iloc[split_index:].copy()

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("\nTraining records:", len(train_data))
print("Testing records :", len(test_data))

print(
    "\nTraining period:",
    train_data["time"].iloc[0],
    "to",
    train_data["time"].iloc[-1]
)

print(
    "Testing period :",
    test_data["time"].iloc[0],
    "to",
    test_data["time"].iloc[-1]
)


# ==============================
# SAVE CLEAN DATA
# ==============================

train_data.to_csv(
    "real_train_prices.csv",
    index=False
)

test_data.to_csv(
    "real_test_prices.csv",
    index=False
)

print("\n" + "=" * 60)
print("FILES CREATED")
print("=" * 60)

print("\n1. real_train_prices.csv")
print("2. real_test_prices.csv")

print("\nReal dataset preparation completed successfully!")