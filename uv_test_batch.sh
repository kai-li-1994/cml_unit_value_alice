#!/bin/bash -l
#SBATCH --job-name=uv_test
#SBATCH --error=logs/uv_test_%j.err
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=4G

module purge
module load ALICE/default
module load Miniconda3/24.7.1-0
module load R/4.4.0-gfbf-2023a

export MY_ENV=alice
export R_LIBS_USER=/zfsstore/user/lik6/R/x86_64-pc-linux-gnu-library/4.4

conda activate cml_uv

cd /zfsstore/user/lik6/cml_unit_value

CHUNK_FILE=task_chunks/2010/task_chunk_00000.csv

echo "Testing on $CHUNK_FILE"

python main.py --chunk $CHUNK_FILE
