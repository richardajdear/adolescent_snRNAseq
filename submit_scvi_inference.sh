#!/bin/bash
#SBATCH -A VERTES-SL2-GPU
#SBATCH -p ampere                      # Target Ampere GPU partition
#SBATCH -J scvi_inference              # Job name for scVI inference
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                   # Request 1 GPU (inference often uses only one)
#SBATCH --time=02:00:00                # Adjust time based on expected runtime
#SBATCH --output=scvi_inference_job_%j.out # Standard output and error log
#SBATCH --error=scvi_inference_job_%j.err  # Standard error log

# --- Singularity Container Configuration ---
SINGULARITY_IMAGE="shortcake_scvi.sif" # !! IMPORTANT: Ensure this path is correct !!

# --- Input Paths ---
# Adjust these paths to your actual files on the HPC
ADATA_PATH="/home/rajd2/rds/hpc-work/snRNAseq/outputs/velmeshev-wang_pcs20/velmeshev-wang_pcs20.h5ad"
MODEL_PATH="/home/rajd2/rds/hpc-work/snRNAseq/outputs/velmeshev-wang_pcs20/scvi_model/model.pt"

# --- Inference Parameters ---
N_SAMPLES=10
TRANSFORM_BATCH="multiome" # Or other specific batch key if desired
MODEL_TYPE="scvi"          # Or "scanvi" if loading a scANVI model
TARGET_LAYER_NAME="scvi_normalized_inference" # Name for the new layer in the AnnData object

echo "--- Starting scVI/scANVI Inference Job ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Singularity image: $SINGULARITY_IMAGE"
echo "Requested partition: $SLURM_JOB_PARTITION"
echo "Requested GPUs: $SLURM_GPUS"

# --- Debugging Steps for GPU/CUDA Configuration (optional but good to keep) ---
echo -e "\n--- Host NVIDIA Driver Information (outside container) ---"
nvidia-smi || echo "WARNING: nvidia-smi command failed. Ensure NVIDIA drivers are installed on the host or in PATH."

echo -e "\n--- Container's CUDA and PyTorch Information (inside container) ---"
singularity exec --nv "$SINGULARITY_IMAGE" bash -c "which nvcc && nvcc --version || echo 'INFO: nvcc not found or command failed in container. This is common if only runtime libraries are present.'"
singularity exec --nv "$SINGULARITY_IMAGE" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    python -c 'import torch; print(f"torch.__version__: {torch.__version__}"); print(f"PyTorch CUDA available: {torch.cuda.is_available()}"); print(f"PyTorch CUDA version: {torch.version.cuda}"); print(f"PyTorch cuDNN version: {torch.backends.cudnn.version()}")' \
    || echo "WARNING: Failed to query PyTorch CUDA information. Ensure PyTorch is installed correctly in the environment."

echo -e "\n--- Running scVI/scANVI Inference Script ---"

# Execute the Python script, passing arguments
singularity exec --nv \
    "$SINGULARITY_IMAGE" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    python snRNAseq/code/run_scvi_inference.py \
        --adata_path "$ADATA_PATH" \
        --model_path "$MODEL_PATH" \
        --n_samples "$N_SAMPLES" \
        --transform_batch "$TRANSFORM_BATCH" \
        --model_type "$MODEL_TYPE" \
        --target_layer_name "$TARGET_LAYER_NAME" \
        --verbose

EXIT_CODE=$?

echo -e "\n--- Job Finished ---"
if [ $EXIT_CODE -eq 0 ]; then
    echo "scVI/scANVI Inference job completed successfully."
else
    echo "scVI/scANVI Inference job failed with exit code $EXIT_CODE."
fi

exit $EXIT_CODE