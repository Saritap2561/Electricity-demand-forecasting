
import pandas as pd

print("Loading electricity demand dataset...")

df = pd.read_csv("AEP_hourly.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime")

print("\n--- DATASET INFORMATION ---")
print("Total records:", len(df))
print("First date:", df["Datetime"].min())
print("Last date:", df["Datetime"].max())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DEMAND STATISTICS (MW) ---")
print(df["AEP_MW"].describe())

print("\n--- FIRST 5 RECORDS ---")
print(df.head())

print("\nDataset inspection completed!")
