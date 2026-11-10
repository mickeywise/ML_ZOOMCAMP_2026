from pathlib import Path
from zipfile import ZipFile, is_zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

ZIP_PATH = RAW_DATA_DIR / "beijing_air_quality.zip"
EXTRACT_DIR = RAW_DATA_DIR / "beijing_air_quality"

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"Dataset ZIP not found: {ZIP_PATH}")

if not is_zipfile(ZIP_PATH):
    raise ValueError("The downloaded file is not a valid ZIP archive.")

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

# Extract the outer ZIP archive
with ZipFile(ZIP_PATH, "r") as archive:
    archive.extractall(EXTRACT_DIR)

print("Outer archive extracted successfully.")

# Find and extract any nested ZIP archives
nested_zips = list(EXTRACT_DIR.rglob("*.zip"))

for nested_zip in nested_zips:
    nested_extract_dir = nested_zip.parent / nested_zip.stem
    nested_extract_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(nested_zip, "r") as archive:
        archive.extractall(nested_extract_dir)

    print(f"Extracted nested archive: {nested_zip.name}")

# Find all station CSV files
csv_files = list(EXTRACT_DIR.rglob("PRSA_Data_*.csv"))
print("\nExtraction complete.")
print(f"CSV files found: {len(csv_files)}")

for file in csv_files:
    print(file.name)

if len(csv_files) != 12:
    print("WARNING: Expected 12 station CSV files.")