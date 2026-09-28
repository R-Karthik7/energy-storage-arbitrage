import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_FILE = "energy_dataset.csv"

TRAIN_SIZE = 28051
TEST_SIZE = 7013


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

print("=" * 65)
print("PREPARING ACTUAL-PRICE SCENARIO")
print("=" * 65)

df = pd.read_csv(
    SOURCE_FILE
)

print(
    f"\nOriginal records: {len(df)}"
)

print(
    f"Original columns: {len(df.columns)}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "time",
    "price day ahead",
    "price actual"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column not found: {column}"
        )


# ============================================================
# CLEAN TIME COLUMN
# ============================================================

df["time"] = pd.to_datetime(
    df["time"],
    utc=True
)

df = df.sort_values(
    "time"
).reset_index(
    drop=True
)


# ============================================================
# CHECK PRICE COLUMNS
# ============================================================

df = df[
    [
        "time",
        "price day ahead",
        "price actual"
    ]
].copy()


df["price day ahead"] = pd.to_numeric(
    df["price day ahead"],
    errors="coerce"
)

df["price actual"] = pd.to_numeric(
    df["price actual"],
    errors="coerce"
)


# Remove rows where either price is missing

df = df.dropna(
    subset=[
        "price day ahead",
        "price actual"
    ]
).reset_index(
    drop=True
)


print(
    f"Clean records: {len(df)}"
)


# ============================================================
# VERIFY EXPECTED SIZE
# ============================================================

if len(df) != TRAIN_SIZE + TEST_SIZE:

    raise ValueError(
        "Cleaned dataset size does not match "
        f"expected {TRAIN_SIZE + TEST_SIZE} records."
    )


# ============================================================
# SPLIT USING SAME 80/20 BOUNDARY
# ============================================================

train_df = df.iloc[
    :TRAIN_SIZE
].copy()

test_df = df.iloc[
    TRAIN_SIZE:
].copy()


# ============================================================
# CREATE ACTUAL-PRICE FILES
# ============================================================

train_actual = train_df[
    [
        "time",
        "price actual"
    ]
].copy()

test_actual = test_df[
    [
        "time",
        "price actual"
    ]
].copy()


train_actual.to_csv(
    "real_train_actual_prices.csv",
    index=False
)

test_actual.to_csv(
    "real_test_actual_prices.csv",
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 65)
print("ACTUAL-PRICE DATA PREPARATION COMPLETED")
print("=" * 65)

print(
    f"\nTraining records: "
    f"{len(train_actual)}"
)

print(
    f"Test records: "
    f"{len(test_actual)}"
)

print(
    f"\nActual price minimum: "
    f"€{df['price actual'].min():.2f}/MWh"
)

print(
    f"Actual price maximum: "
    f"€{df['price actual'].max():.2f}/MWh"
)

print(
    f"Actual price average: "
    f"€{df['price actual'].mean():.2f}/MWh"
)

print("\nCreated files:")

print(
    "1. real_train_actual_prices.csv"
)

print(
    "2. real_test_actual_prices.csv"
)

print("\n" + "=" * 65)