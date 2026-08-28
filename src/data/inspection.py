import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "nasa_power_tunis.csv"


df = pd.read_csv(INPUT_PATH)

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== LAST 5 ROWS =====")
print(df.tail())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== BASIC STATISTICS =====")
print(df.describe())