import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Locate project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input dataset
DATA_PATH = PROJECT_ROOT / "sales_forecast" / "data" / "SuperKart (1).csv"

# Output files expected by train.py and GitHub Actions
OUTPUT_DIR = PROJECT_ROOT

# Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)

# Remove identifier column
df.drop(columns=["Product_Id"], inplace=True)

# Define target
target_col = "Product_Store_Sales_Total"

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found.")

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Save splits
Xtrain.to_csv(OUTPUT_DIR / "Xtrain.csv", index=False)
Xtest.to_csv(OUTPUT_DIR / "Xtest.csv", index=False)
ytrain.to_frame(name=target_col).to_csv(
    OUTPUT_DIR / "ytrain.csv", index=False
)
ytest.to_frame(name=target_col).to_csv(
    OUTPUT_DIR / "ytest.csv", index=False
)

print("Train-test split completed.")
print("Xtrain:", Xtrain.shape)
print("Xtest:", Xtest.shape)
print("ytrain:", ytrain.shape)
print("ytest:", ytest.shape)
print("All four CSV files saved in:", OUTPUT_DIR)
