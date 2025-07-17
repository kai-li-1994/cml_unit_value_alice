#!/usr/bin/env python3
import pandas as pd
import glob
import os

def main():
    reports_dir = '/zfsstore/user/lik6/cml_unit_value/reports'  # Change as needed
    parquet_files = sorted(glob.glob(os.path.join(reports_dir, '*.parquet')))
    if not parquet_files:
        print(f"No .parquet files found in {reports_dir}")
        return

    dfs = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            row = df['value'].to_dict()
            row['source_file'] = os.path.basename(f)
            dfs.append(pd.DataFrame([row]))
        except Exception as e:
            print(f"Failed to read {f}: {e}")

    if not dfs:
        print("No files read successfully.")
        return

    df_combined = pd.concat(dfs, ignore_index=True)
    combined_csv = os.path.join(reports_dir, 'combined_report.csv')
    df_combined.to_csv(combined_csv, index=False)
    print(f"Combined DataFrame shape: {df_combined.shape}")
    print(f"Saved combined_report.csv in {reports_dir}")

if __name__ == "__main__":
    main()

