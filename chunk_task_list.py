import os
import pandas as pd
import argparse

def chunk_task_list(
    years=None, 
    chunk_size=70, 
    input_csv='master_task_list.csv',
    output_prefix='task_chunk', 
    output_dir='task_chunks'
):
    """Chunk the master task list by years and chunk size, with proper HS code formatting."""
    # Always read hs_code as string
    df = pd.read_csv(input_csv, dtype={'hs_code': str})
    df['hs_code'] = df['hs_code'].str.zfill(6)  # Zero-pad all codes

    if years:
        df = df[df['year'].isin(years)]
    df = df.reset_index(drop=True)
    print(f"Total rows after filtering: {len(df)}")

    # Make output directory if not exist
    os.makedirs(output_dir, exist_ok=True)

    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        chunk = df.iloc[i*chunk_size:(i+1)*chunk_size].copy()
        chunk['hs_code'] = chunk['hs_code'].str.zfill(6)  # Ensure each chunk has correct format
        chunk_file = os.path.join(output_dir, f"{output_prefix}_{i:05d}.csv")
        chunk.to_csv(chunk_file, index=False)
        print(f"Saved {chunk_file} with {len(chunk)} rows.")
    print(f"Done! {n_chunks} chunk files created in {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunk master_task_list.csv into smaller files for batch processing."
    )
    parser.add_argument('--years', type=str, default='all',
                        help="Comma-separated years (e.g. 2010,2011) or 'all' [default: all]")
    parser.add_argument('--chunk-size', type=int, default=70,
                        help="Number of rows per chunk [default: 70]")
    parser.add_argument('--input-csv', type=str, default='master_task_list.csv',
                        help="Input CSV file [default: master_task_list.csv]")
    parser.add_argument('--output-prefix', type=str, default='task_chunk',
                        help="Output chunk file prefix [default: task_chunk]")
    parser.add_argument('--output-dir', type=str, default='task_chunks',
                        help="Output directory for chunk files [default: task_chunks]")

    args = parser.parse_args()

    if args.years == 'all':
        years = None
    else:
        years = [int(y) for y in args.years.split(',')]

    chunk_task_list(
        years=years,
        chunk_size=args.chunk_size,
        input_csv=args.input_csv,
        output_prefix=args.output_prefix,
        output_dir=args.output_dir
    )
