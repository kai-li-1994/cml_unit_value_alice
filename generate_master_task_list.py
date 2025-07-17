import os
import re
import pandas as pd

base_dir = 'data_uncomtrade'
years = [
    2010, 2011, 2012, 2013, 2014, 2015, 2016,
    2017, 2018, 2019, 2020, 2021, 2022, 2023
]
flows = ['m', 'x']

task_rows = []

for year in years:
    subfolder = f'split_by_hs_{year}_numpy'
    year_dir = os.path.join(base_dir, subfolder)
    if not os.path.exists(year_dir):
        print(f"Warning: {year_dir} does not exist, skipping.")
        continue
    for filename in os.listdir(year_dir):
        # Exclude files with 'unknown' in the name and summary/logs
        if not (filename.endswith('.csv') and 'unknown' not in filename and 'summary' not in filename and 'log' not in filename):
            continue
        # Extract HS code with a regex (handles any prefix)
        m = re.match(r'[A-Za-z0-9]+_([0-9]{6})_' + str(year) + r'\.csv', filename)
        if m:
            hs_code = m.group(1)
            for flow in flows:
                task_rows.append({'hs_code': hs_code, 'year': year, 'flow': flow})

df_tasks = pd.DataFrame(task_rows)
df_tasks = df_tasks.sort_values(['year', 'hs_code', 'flow'])
df_tasks.to_csv('master_task_list.csv', index=False)
print(f"Saved master_task_list.csv with {len(df_tasks)} rows.")
