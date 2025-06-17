#!/bin/bash
#SBATCH -A VERTES-SL2-CPU
#SBATCH -p cclake-himem
#SBATCH -J metacell_agg                     # Job name (changed from integrate)
#SBATCH --nodes=1                           # Request 1 node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=08:00:00                     # Set a maximum run time (HH:MM:SS)
#SBATCH --output=metacell_agg_job_%j.out    # Standard output and error log (%j expands to jobID)
#SBATCH --error=metacell_agg_job_%j.err     # Standard error log

# --- Singularity Container Configuration ---
SINGULARITY_IMAGE="/home/rajd2/rds/hpc-work/singularity_images/shortcake.sif" # IMPORTANT: Update this path to your actual image!
# Assuming 'shortcake.sif' contains your micromamba environment with scanpy, pandas, numpy, etc.
# If your environment is named differently, adjust `micromamba run -n your_env_name` below.

# --- Paths for your Python Script and Data ---
# Full path to your aggregation Python script on the host machine
PYTHON_SCRIPT_PATH="/home/rajd2/rds/hpc-work/snRNAseq/code/aggregate_metacells.py"

# Full path to your original AnnData file (e.g., the output from your Harmony integration script)
# This file should contain the raw counts (in .X or a layer) and the obs metadata (donor_age, cell_lineage, etc.)
ORIGINAL_ADATA_FILE="/home/rajd2/rds/hpc-work/snRNAseq/outputs/velmeshev-wang_pcs20/velmeshev-wang_pcs20.h5ad"

# Full path to the MetaQ-generated metacell assignments file (_ids.h5ad)
METACELL_ASSIGNMENTS_FILE="/home/rajd2/rds/hpc-work/snRNAseq/outputs/velmeshev-wang_pcs20/metaq/save/velmeshev-wang_pcs20_Xharmony_10000metacell_ids.h5ad"

# --- Aggregation Specific Parameters ---
# Set to 'counts' if raw counts are in adata.layers['counts'], 'raw_counts' if in adata.layers['raw_counts'],
# or 'None' (as a string) if raw counts are directly in adata.X.
RAW_COUNTS_LAYER_KEY="None" # Or "counts", "raw_counts", etc.

# List of quantitative .obs columns to aggregate (comma-separated, no spaces)
# Example: "age_years,age_log2,total_counts"
QUANTITATIVE_COLS="age_years,age_log2,total_counts,n_genes_by_counts"

# List of categorical .obs columns to aggregate (comma-separated, no spaces)
# Example: "origin,dataset,lineage"
CATEGORICAL_COLS="origin,dataset,chemistry,lineage,region,tissue,donor_id,sex,development_stage"


# --- Execute the Python Script via Singularity ---
# The micromamba environment name might need to be adjusted if it's not 'shortcake_default'
# The script will automatically determine the output path.
singularity exec --cleanenv \
                 -B "/home/rajd2/rds/hpc-work:/home/rajd2/rds/hpc-work" \
                 "${SINGULARITY_IMAGE}" \
                 micromamba run -n shortcake_default \
                 python "${PYTHON_SCRIPT_PATH}" \
                 --original_adata_path "${ORIGINAL_ADATA_FILE}" \
                 --metacell_assignments_path "${METACELL_ASSIGNMENTS_FILE}" \
                 --raw_counts_layer_key "${RAW_COUNTS_LAYER_KEY}" \
                 --quantitative_obs_cols "${QUANTITATIVE_COLS}" \
                 --categorical_obs_cols "${CATEGORICAL_COLS}"