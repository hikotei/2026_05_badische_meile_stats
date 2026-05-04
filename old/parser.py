import pandas as pd


def parse_race_results(html_file_path):
    print(f"Reading tables from {html_file_path}...")

    # 1. Read the HTML file
    # Pandas automatically finds all <table> tags and returns a list of DataFrames
    tables = pd.read_html(html_file_path)

    # 2. Select your table
    # Assuming the main results table is the first one in the document
    df = tables[0]

    print(f"Found table with {len(df)} rows and {len(df.columns)} columns.")

    # 3. Clean the data
    # Your HTML snippet shows the first and last columns are empty (comments/favorites).
    # Pandas usually names these empty headers "Unnamed: 0", "Unnamed: 9", etc.
    cols_to_drop = [col for col in df.columns if "Unnamed" in str(col)]
    df_cleaned = df.drop(columns=cols_to_drop)

    # Clean up string columns to remove extra whitespace or newline characters
    # (Sometimes HTML formatting leaves hidden spacing)
    for col in df_cleaned.select_dtypes(include=["object"]).columns:
        df_cleaned[col] = df_cleaned[col].str.strip()

    # 4. Export to CSV (Most universal)
    csv_filename = "race_results.csv"
    df_cleaned.to_csv(csv_filename, index=False, encoding="utf-8")
    print(f"Successfully saved to {csv_filename}")

    # 5. Export to Parquet (Most efficient for storage/speed)
    # Note: Requires the 'pyarrow' or 'fastparquet' library installed
    parquet_filename = "race_results.parquet"
    try:
        df_cleaned.to_parquet(parquet_filename, index=False)
        print(f"Successfully saved to {parquet_filename}")
    except ImportError:
        print("Tip: Install 'pyarrow' (pip install pyarrow) to enable Parquet exports.")


# Run the script
if __name__ == "__main__":
    # Replace 'results.html' with the actual path to your file
    parse_race_results("website.html")
