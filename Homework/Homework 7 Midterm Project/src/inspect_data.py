from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "beijing_air_quality"

csv_files = list(DATA_DIR.rglob("*.csv"))

for file in csv_files:
    print("\n" + "=" * 70)
    print(f"FILE: {file.name}")
    print("=" * 70)

    df = pd.read_csv(file)

    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head().to_string())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())