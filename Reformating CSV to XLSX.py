import os
import pandas as pd
from multiprocessing import Pool

# Directory containing CSV files
csv_directory = r"C:"

# Output directory for XLSX files
xlsx_directory = r"E:"

# Create the output directory if it doesn't exist
os.makedirs(xlsx_directory, exist_ok=True)


def convert_csv_to_xlsx(filename):
    if filename.endswith(".csv"):
        # Construct the full paths
        csv_file = os.path.join(csv_directory, filename)
        xlsx_file = os.path.join(xlsx_directory, filename.replace(".csv", ".xlsx"))

        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Save as XLSX
        df.to_excel(xlsx_file, index=False)

        print(f"Converted: {filename}")


if __name__ == "__main__":
    csv_files = [filename for filename in os.listdir(csv_directory) if filename.endswith(".csv")]

    # Use multiple processes to convert CSV files to XLSX in parallel
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(convert_csv_to_xlsx, csv_files)

    print("Conversion complete.")

