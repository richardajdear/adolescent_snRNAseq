#!/bin/bash
#SBATCH -A VERTES-SL2-GPU
#SBATCH -p ampere
#SBATCH -J metaq                            # Job name
#SBATCH --nodes=1                           # Request 1 node
#SBATCH --gres=gpu:2                        
#SBATCH --time=02:00:00                     # Set a maximum run time (HH:MM:SS)
#SBATCH --output=metaq_job_%j.out           # Standard output and error log (%j expands to jobID)
#SBATCH --error=metaq_job_%j.err            # Standard error log


### RUN SBATCH FROM THE /metaq DIRECTORY FOR THE GIVEN DATASET ###
# e.g. cd /home/rajd2/rds/hpc-work/snRNAseq/outputs/velmeshev100k-wang50k_pcs20/metaq

# --- Define Input Parameters (relative to job submission directory) ---
# Data file path relative to /metaq directory 
# Make sure to prepend ../ if your job is submitted from '.../metaq/' and your data file is '.../velmeshev100k-wang50k_pcs20_Xharmony.h5ad'
DATA_FILE_RELATIVE="../velmeshev-wang_pcs20.h5ad" # This looks correct for the assumed structure

METACELL_NUM=10000
DATA_TYPE="RNA"


# --- Define Global Paths ---
# Predefined path to your Singularity image
SINGULARITY_IMAGE="/home/rajd2/rds/hpc-work/metaq.sif"

# Path to your patched MetaQ_sc files on the host machine
PATCH_DIR="/home/rajd2/rds/hpc-work/snRNAseq/metaq/patches/MetaQ_sc"

# Target path inside the container where MetaQ_sc is installed
CONTAINER_METAQ_PATH="/opt/conda/envs/metaq_env/lib/python3.11/site-packages/MetaQ_sc"

# Full path to your Python script
PYTHON_SCRIPT_PATH="/home/rajd2/rds/hpc-work/snRNAseq/code/run_metaq.py"


# --- Execute the Singularity container and Python script ---
singularity exec --nv \
                 --cleanenv \
                 -B "${PATCH_DIR}:${CONTAINER_METAQ_PATH}" \
                 "${SINGULARITY_IMAGE}" \
                 micromamba run -n metaq_env \
                 python "${PYTHON_SCRIPT_PATH}" \
                 --data_file "${DATA_FILE_RELATIVE}" \
                 --metacell_num "${METACELL_NUM}" \
                 --data_type "${DATA_TYPE}" \
                 --device "cuda"