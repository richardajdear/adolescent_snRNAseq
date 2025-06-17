#!/bin/bash
#SBATCH -A VERTES-SL2-GPU               # Your GPU account
#SBATCH -p ampere                       # Target Ampere GPU partition (e.g., volta-gpu, turing-gpu, ampere)
#SBATCH -J scanvi                       # Job name for scVI
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --gres=gpu:2                    
#SBATCH --time=00:30:00                 # Set a maximum run time (HH:MM:SS) - adjust if actual run takes longer
#SBATCH --output=scvi_job_%j.out        # Standard output and error log (%j expands to jobID)
#SBATCH --error=scvi_job_%j.err         # Standard error log

# Set name for the run, which defines where to find the harmony dataset 
RUN_NAME="velmeshev100k-wang50k_pcs20"
# RUN_NAME="velmeshev-wang_pcs20"

# --- Singularity Container Configuration ---
SINGULARITY_IMAGE="shortcake_scvi.sif" # !! IMPORTANT: Ensure this path is correct relative to job submission !!

echo "--- Starting scVI/scANVI Integration Job ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Singularity image: $SINGULARITY_IMAGE"
echo "Requested partition: $SLURM_JOB_PARTITION"
echo "Requested GPUs: $SLURM_GPUS"

# --- Debugging Steps for GPU/CUDA Configuration ---
echo -e "\n--- Host NVIDIA Driver Information (outside container) ---"
# This command shows the NVIDIA driver version on the host.
# Critical for checking compatibility with CUDA versions inside the container.
nvidia-smi || echo "WARNING: nvidia-smi command failed. Ensure NVIDIA drivers are installed on the host or in PATH."

echo -e "\n--- Container's CUDA and PyTorch Information (inside container) ---"

# Check CUDA Toolkit version inside the container (if nvcc is installed and in PATH)
echo "Attempting to check CUDA Toolkit version inside container (if nvcc is present):"
singularity exec --nv "$SINGULARITY_IMAGE" bash -c "which nvcc && nvcc --version || echo 'INFO: nvcc not found or command failed in container. This is common if only runtime libraries are present.'"

# Check PyTorch's CUDA status and the CUDA version it was compiled against
# This is the most critical diagnostic step for 'Error 803'.
echo "Checking PyTorch CUDA status and versions within environment 'scvi-scgen-scmomat-unitvelo':"
singularity exec --nv "$SINGULARITY_IMAGE" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    python -c 'import torch; print(f"torch.__version__: {torch.__version__}"); print(f"PyTorch CUDA available: {torch.cuda.is_available()}"); print(f"PyTorch CUDA version: {torch.version.cuda}"); print(f"PyTorch cuDNN version: {torch.backends.cudnn.version()}")' \
    || echo "WARNING: Failed to query PyTorch CUDA information. Ensure PyTorch is installed correctly in the environment."

echo -e "\n--- Running scVI/scANVI Integration Script ---"

# Execute the main Python script
# Make sure 'snRNAseq/code/run_scvi.py' is the correct path to your script
singularity exec --nv \
    "$SINGULARITY_IMAGE" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    python snRNAseq/code/run_scvi.py $RUN_NAME \
    --num_gpus_for_scanvi_training 1 \
    --run_inference

# --overwrite_scanvi_model \

EXIT_CODE=$?

echo -e "\n--- Job Finished ---"
if [ $EXIT_CODE -eq 0 ]; then
    echo "scVI/scANVI Integration job completed successfully."
else
    echo "scVI/scANVI Integration job failed with exit code $EXIT_CODE."
fi

exit $EXIT_CODE