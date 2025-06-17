#!/bin/bash
#SBATCH -A VERTES-SL2-CPU
#SBATCH -p cclake-himem
#SBATCH -J integrate                        # Job name
#SBATCH --nodes=1                           # Request 1 node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=08:00:00                     # Set a maximum run time (HH:MM:SS)
#SBATCH --output=harmony_job_%j.out         # Standard output and error log (%j expands to jobID)
#SBATCH --error=harmony_job_%j.err          # Standard error log

# --- Singularity Container Configuration ---
SINGULARITY_IMAGE="shortcake.sif" # !! IMPORTANT: Update this path !!

echo "Starting Harmony Integration job using Singularity container: $SINGULARITY_IMAGE"

# Use 'singularity exec' and directly specify the Python executable path
# within the micromamba environment inside the container.
# Ensure this path is correct for your 'shortcake_default' environment.
# You might need to adjust '/opt/micromamba/envs/shortcake_default/bin/python'
# based on the actual path in your SIF.
srun singularity exec \
    "$SINGULARITY_IMAGE" \
    bash -c "export PYTHONPATH=\$PYTHONPATH:$(pwd)/snRNAseq/code && \
             /opt/micromamba/envs/shortcake_default/bin/python snRNAseq/code/run_harmony.py"

echo "Harmony Integration job finished."