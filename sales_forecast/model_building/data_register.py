import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "sales_forecast" / "data" / "SuperKart (1).csv"

if not RAW_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

df = pd.read_csv(RAW_PATH)

expected_columns = [
    "Product_Id",
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Store_Sales_Total",
]

missing = [column for column in expected_columns if column not in df.columns]

if missing:
    raise ValueError(f"Missing expected columns: {missing}")

print("SuperKart dataset registered successfully.")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print("Target column: Product_Store_Sales_Total")
