import scanpy as sc
import scanpy.external as sce
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings
import torch
import argparse
from typing import Optional
import time # Import the time module
import torch.distributed as dist # Import torch.distributed
from lightning.pytorch.callbacks import EarlyStopping # Only import EarlyStopping
import shutil # Import shutil for robust directory removal

# Determine local rank for controlled printing in DDP
# IMPORTANT: This will be 0 if not launched via torchrun/srun --nodes=1 --ntasks-per-node=N (where N > 1)
# or if running as a single-process job.
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
# Get world size and rank from environment variables set by the launcher
GLOBAL_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
GLOBAL_RANK = int(os.environ.get("RANK", 0))

# Import the new plotting utility function
sys.path.append(os.path.join(os.path.dirname(__file__))) # Add current directory to path
try:
    from plot_utils import plot_integration_umaps
except ImportError:
    if LOCAL_RANK == 0: # Only print error from main process
        print("Error: Could not import plot_integration_umaps from plot_utils.py.")
        print("Please ensure 'plot_utils.py' is in the 'snRNAseq/code' directory.")
    sys.exit(1)

# Import the new inference utility
try:
    from scvi_inference_utils import get_normalized_expression_chunked
except ImportError:
    if LOCAL_RANK == 0: # Only print error from main process
        print("Error: Could not import get_normalized_expression_chunked from scvi_inference_utils.py.")
        print("Please ensure 'scvi_inference_utils.py' is in the 'snRNAseq/code' directory.")
    sys.exit(1)

try:
    import scvi
    from scvi.model import SCVI, SCANVI
except ImportError:
    if LOCAL_RANK == 0: # Only print error from main process
        print("Error: Could not import scvi. Please ensure it is installed (`pip install scvi-tools`).")
        print("If running on a GPU cluster, ensure scvi-tools is installed in your GPU environment.")
    sys.exit(1)


def integrate_snrnaseq_scvi(
    adata: ad.AnnData,
    output_dir: str = "snRNAseq/combined",
    output_filename: str = "integrated_scvi.h5ad",
    scvi_model_path: str = "scvi_model",
    scanvi_model_path: str = "scanvi_model",
    n_latent: int = 30,
    n_hidden: int = 128,
    n_layers: int = 2,
    max_epochs_scvi: int = 400,
    max_epochs_scanvi: int = 20,
    batch_key: str = 'chemistry',
    cell_type_key: str = 'lineage',
    n_samples_normalize_expression: int = 10, # Changed default from 1 to 10
    transform_batch_normalize_expression: str = 'multiome',
    plot_umaps: bool = True,
    plot_color_keys: list[str] = None,
    figure_size_inches: tuple[int, int] = (20, 10), # Default provided
    age_log2_plot_settings: dict = None,
    random_state: int = 42,
    use_highly_variable_genes: bool = True,
    n_neighbors_umap: int = 30,
    min_dist_umap: float = 0.3,
    num_workers_scvi: int = 8,
    chunk_size_normalize_expression: Optional[int] = None,
    num_gpus_for_training: Optional[int] = None, # Number of GPUs for SCVI training
    num_gpus_for_scanvi_training: Optional[int] = 1, # Number of GPUs for scANVI training (defaults to 1, will be non-DDP)
    overwrite_scvi_model: bool = False, # If True, train SCVI regardless of existing model
    overwrite_scanvi_model: bool = False, # If True, train scANVI regardless of existing model
    run_inference: bool = False, # If False, skip generating normalized expression layers
) -> ad.AnnData:
    """
    Integrates single-nucleus RNA-seq (snRNA-seq) data using scVI and scANVI
    for batch correction and dimensionality reduction.

    This function takes an AnnData object (presumably preprocessed and with raw counts
    stored in a 'counts' layer), sets up scVI and scANVI models, trains them,
    and stores the latent representations and batch-corrected normalized data
    in the AnnData object. It also generates UMAP plots.

    Args:
        adata: An AnnData object containing the raw counts (expected in .layers['counts'])
               and highly variable genes identified (expected in .var['highly_variable']).
        output_dir: Directory to save the integrated AnnData object and plots.
        output_filename: Filename for the integrated AnnData object.
        scvi_model_path: Directory path to save/load the trained SCVI model.
        scanvi_model_path: Directory path to save/load the trained scANVI model.
        n_latent: Dimensionality of the latent space for scVI/scANVI.
        n_hidden: Number of nodes per hidden layer in the neural networks.
        n_layers: Number of hidden layers in the neural networks.
        max_epochs_scvi: Maximum number of epochs for training the SCVI model.
        max_epochs_scanvi: Maximum number of epochs for training the scANVI model.
        batch_key: The observation key in AnnData (.obs) that identifies the batch.
        cell_type_key: The observation key in AnnData (.obs) that identifies cell types
                       for scANVI's semi-supervised learning.
        n_samples_normalize_expression: Number of Monte Carlo samples to draw for `get_normalized_expression`.
        transform_batch_normalize_expression: The `transform_batch` parameter for `get_normalized_expression`.
                                              Defaults to 'multiome'.
        plot_umaps: If True, generates and displays UMAP plots of the integrated data.
        plot_color_keys: A list of .obs keys to use for coloring the UMAP plots.
                         Defaults to ['origin', 'dataset', 'chemistry', 'age_log2', 'lineage', 'tissue'].
        figure_size_inches: Tuple specifying the width and height of the UMAP plot figure.
        age_log2_plot_settings: Dictionary containing settings for 'age_log2' colorbar.
                                Expected keys: 'age_ticks' (np.array). If None, default settings will be used.
        random_state: Random seed for reproducibility.
        use_highly_variable_genes: If True, models will be trained only on highly variable genes
                                   (as indicated by `adata.var['highly_variable']`).
                                   If False, models will be trained on all genes.
        n_neighbors_umap: Number of neighbors for UMAP graph construction.
        min_dist_umap: Minimum distance parameter for UMAP.
        num_workers_scvi: Number of data loader workers for scVI and scANVI training.
                          Should not exceed the `--cpus-per-task` in your SLURM script.
        chunk_size_normalize_expression: Optional. The number of cells to process in each chunk
                                         when generating normalized expression. If None, it will be
                                         estimated based on available GPU VRAM.
        num_gpus_for_training: Optional. The number of GPUs to use for model training (SCVI).
        num_gpus_for_scanvi_training: Optional. The number of GPUs to use for scANVI training.
                                      Defaults to 1 to avoid DDP issues if SCVI was multi-GPU.
        overwrite_scvi_model: If True, SCVI model will be trained and overwritten regardless of existing model.
        overwrite_scanvi_model: If True, scANVI model will be trained and overwritten regardless of existing model.
        run_inference: If False, skip generating normalized expression layers.

    Returns:
        An AnnData object containing the integrated and processed data with scVI/scANVI embeddings.
    """
    if plot_color_keys is None:
        plot_color_keys = ['origin', 'dataset', 'chemistry', 'age_log2', 'lineage', 'tissue']
    if age_log2_plot_settings is None:
        age_log2_plot_settings = {'age_ticks': np.array([0, 1, 3, 5, 9, 25])}

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, scvi_model_path), exist_ok=True)
    os.makedirs(os.path.join(output_dir, scanvi_model_path), exist_ok=True)

    if LOCAL_RANK == 0:
        print("\n--- scVI/scANVI Integration Pipeline ---")
    
    # Initialize start time for this function call
    function_start_time = time.time()

    scvi.settings.seed = random_state
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        if LOCAL_RANK == 0:
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            # Apply Tensor Core optimization
            torch.set_float32_matmul_precision('high')
            print("  Set torch.set_float32_matmul_precision('high') for Tensor Core optimization.")

        # Determine num_gpus for SCVI training
        if num_gpus_for_training is None:
            num_gpus_for_training = 1 # Default to 1 if not specified
            if LOCAL_RANK == 0:
                print(f"  'num_gpus_for_training' not specified, defaulting to {num_gpus_for_training} GPU for training.")
        elif num_gpus_for_training > torch.cuda.device_count():
            if LOCAL_RANK == 0:
                warnings.warn(f"Requested {num_gpus_for_training} GPUs for SCVI training, but only {torch.cuda.device_count()} are available. Using all available GPUs.")
            num_gpus_for_training = torch.cuda.device_count()
        if LOCAL_RANK == 0:
            print(f"  Training will use {num_gpus_for_training} GPUs.")

        # Determine num_gpus for scANVI training (this will be used only by rank 0 for non-DDP training)
        if num_gpus_for_scanvi_training is None:
            num_gpus_for_scanvi_training = 1 # Default to 1 if not specified
            if LOCAL_RANK == 0:
                print(f"  'num_gpus_for_scanvi_training' not specified, defaulting to {num_gpus_for_scanvi_training} GPU for scANVI training.")
        elif num_gpus_for_scanvi_training > torch.cuda.device_count():
            if LOCAL_RANK == 0:
                warnings.warn(f"Requested {num_gpus_for_scanvi_training} GPUs for scANVI training, but only {torch.cuda.device_count()} are available. Using all available GPUs.")
            num_gpus_for_scanvi_training = torch.cuda.device_count()
        if LOCAL_RANK == 0:
            print(f"  scANVI training will use {num_gpus_for_scanvi_training} GPUs.")

        # Set the device for the current DDP process (for SCVI training and distributed inference)
        torch.cuda.set_device(LOCAL_RANK) # Each process gets its own device if using DDP
    else:
        if LOCAL_RANK == 0:
            print("CUDA not available. Using CPU.")
        num_gpus_for_training = 0 # No GPUs if CUDA is not available
        num_gpus_for_scanvi_training = 0 # No GPUs if CUDA is not available

    # Set scvi-tools number of workers
    scvi.settings.num_workers = num_workers_scvi
    if LOCAL_RANK == 0:
        print(f"  scvi.settings.num_workers set to {scvi.settings.num_workers}.")
    
    # Set global data loader kwargs for drop_last
    # This will apply to AnnDataLoader and SemiSupervisedDataLoader
    scvi.settings.data_loader_cls_kwargs = {"drop_last": True}
    if LOCAL_RANK == 0:
        print("  Set scvi.settings.data_loader_cls_kwargs={'drop_last': True} to prevent batch size 1 errors.")


    # Save the original X_umap (from Harmony) to X_umap_harmony
    if 'X_umap' in adata.obsm:
        adata.obsm['X_umap_harmony'] = adata.obsm['X_umap'].copy()
        if LOCAL_RANK == 0:
            print("  Original 'X_umap' (from Harmony) copied to 'X_umap_harmony' for preservation.")
    else:
        if LOCAL_RANK == 0:
            warnings.warn("No 'X_umap' found in the input AnnData object. "
                          "Harmony UMAP cannot be saved to 'X_umap_harmony'. "
                          "Ensure the input AnnData is from a Harmony integration that computed UMAP.")

    if LOCAL_RANK == 0:
        print("\nStep 1: Preparing AnnData for scVI...")
    if 'counts' not in adata.layers:
        if LOCAL_RANK == 0:
            print("  Storing .X as 'counts' layer for scVI.")
        adata.layers["counts"] = adata.X.copy()
    else:
        if LOCAL_RANK == 0:
            print("  'counts' layer already exists. Using existing raw counts.")

    if use_highly_variable_genes:
        if 'highly_variable' not in adata.var.columns:
            if LOCAL_RANK == 0:
                warnings.warn("`use_highly_variable_genes` is True, but 'highly_variable' not found in .var. "
                              "Training will proceed on all genes. Please ensure highly variable genes are identified "
                              "or set `use_highly_variable_genes=False`.")
            genes_to_use = adata.var_names
            if LOCAL_RANK == 0:
                print("  Training on all genes due to missing 'highly_variable' flag.")
        else:
            genes_to_use = adata.var_names[adata.var['highly_variable']]
            if LOCAL_RANK == 0:
                print(f"  Using {len(genes_to_use)} highly variable genes for training.")
    else:
        genes_to_use = adata.var_names
        if LOCAL_RANK == 0:
            print("  Training on all genes as `use_highly_variable_genes` is False.")

    adata_scvi = adata[:, genes_to_use].copy()
    adata_scvi.X = adata_scvi.layers['counts'].copy()

    if LOCAL_RANK == 0:
        print("  Setting up AnnData for scVI...")
    try:
        SCVI.setup_anndata(adata_scvi, layer="counts", batch_key=batch_key)
        if LOCAL_RANK == 0:
            print("  AnnData setup for scVI complete.")
    except Exception as e:
        raise RuntimeError(f"Error during scVI AnnData setup: {e}. "
                           "Ensure 'counts' layer and batch_key are correctly set.")

    # --- Trainer kwargs for SCVI training ---
    scvi_trainer_kwargs = {
        "accelerator": "gpu",
        "devices": num_gpus_for_training if num_gpus_for_training > 0 else 1,
        "strategy": "ddp_find_unused_parameters_true" if num_gpus_for_training > 1 else "auto",
        "callbacks": [], # Empty callbacks for DDP for now (EarlyStopping added conditionally)
    }
    if num_gpus_for_training == 0: # CPU fallback
        scvi_trainer_kwargs["accelerator"] = "cpu"
        scvi_trainer_kwargs["devices"] = 1
    elif num_gpus_for_training <= 1: # Single GPU or CPU, can use EarlyStopping
        scvi_trainer_kwargs["callbacks"].append(EarlyStopping(monitor="elbo_train", patience=30))


    # --- SCVI Training / Loading Logic ---
    scvi_model_full_path = os.path.join(output_dir, scvi_model_path)
    
    should_train_scvi = overwrite_scvi_model or not os.path.exists(scvi_model_full_path)
    
    if should_train_scvi:
        # Start timer for SCVI training
        scvi_train_start_time = time.time()
        if LOCAL_RANK == 0:
            print(f"\nStep 2: Training SCVI model with {num_gpus_for_training} GPUs...")
        vae = SCVI(adata_scvi, n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers)
        if LOCAL_RANK == 0:
            print(f"  SCVI model initialized with n_latent={n_latent}, n_hidden={n_hidden}, n_layers={n_layers}.")
            print(f"  Training SCVI for {max_epochs_scvi} epochs (using {scvi_trainer_kwargs['accelerator']} and {scvi_trainer_kwargs['devices']} devices)...")
        vae.train(
            max_epochs=max_epochs_scvi,
            early_stopping=False, # Disable scvi-tools internal early stopping
            enable_progress_bar=True, # Keep True for SCVI
            train_size=0.9,
            validation_size=0.1,
            **scvi_trainer_kwargs # Unpack trainer_kwargs here
        )
        if LOCAL_RANK == 0:
            scvi_train_end_time = time.time()
            hours, rem = divmod(scvi_train_end_time - scvi_train_start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"  SCVI model training complete. Elapsed time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
            vae.save(scvi_model_full_path, overwrite=True)
            print(f"  SCVI model saved to {scvi_model_full_path}")
    else:
        if LOCAL_RANK == 0:
            print(f"\nStep 2: Existing SCVI model found at {scvi_model_full_path}. Skipping training and loading model...")
        vae = SCVI.load(scvi_model_full_path, adata=adata_scvi)
        if LOCAL_RANK == 0:
            print("  SCVI model loaded successfully.")
    
    # Ensure model is on the correct device for inference (important if loaded from disk)
    if use_cuda and hasattr(vae, 'module') and hasattr(vae.module, 'to'):
        vae.module.to(torch.device(f"cuda:{LOCAL_RANK}")) # Ensure it's on this rank's device
    elif use_cuda and hasattr(vae, 'to'):
        vae.to(torch.device(f"cuda:{LOCAL_RANK}"))


    # For get_latent_representation, the model's primary device (cuda:0) is used.
    adata.obsm["X_scVI"] = vae.get_latent_representation(adata=adata_scvi)
    if LOCAL_RANK == 0:
        print(f"  SCVI latent representation (X_scVI) stored in adata.obsm with shape {adata.obsm['X_scVI'].shape}.")
    
    # Synchronize all processes after SCVI training/loading before moving to scANVI
    if dist.is_initialized():
        dist.barrier()


    # --- scANVI Training / Loading Logic ---
    scanvi_model_full_path = os.path.join(output_dir, scanvi_model_path)

    should_train_scanvi = overwrite_scanvi_model or not os.path.exists(scanvi_model_full_path)

    # Trainer kwargs for scANVI training - now allows multi-GPU
    scanvi_trainer_kwargs = {
        "accelerator": "gpu",
        "devices": num_gpus_for_scanvi_training if num_gpus_for_scanvi_training > 0 else 1,
        "strategy": "ddp_find_unused_parameters_true" if num_gpus_for_scanvi_training > 1 else "auto", # 'auto' for single GPU
        "callbacks": [], # Empty callbacks for DDP for now (EarlyStopping added conditionally)
    }
    if num_gpus_for_scanvi_training == 0: # CPU fallback
        scanvi_trainer_kwargs["accelerator"] = "cpu"
        scanvi_trainer_kwargs["devices"] = 1
    elif num_gpus_for_scanvi_training <= 1: # Single GPU or CPU, can use EarlyStopping
        scanvi_trainer_kwargs["callbacks"].append(EarlyStopping(monitor="elbo_train", patience=30))


    if should_train_scanvi:
        scANVI_train_start_time = time.time() # Start timer for scANVI training
        if LOCAL_RANK == 0: # Guarded print
            print(f"\nStep 3: Training scANVI model using '{cell_type_key}' as cell type key (using {num_gpus_for_scanvi_training} GPUs)...")
        if cell_type_key not in adata_scvi.obs.columns:
            raise ValueError(f"Cell type key '{cell_type_key}' not found in adata_scvi.obs. "
                               "scANVI requires a cell type annotation for semi-supervised training.")
        if adata_scvi.obs[cell_type_key].isnull().any():
            if LOCAL_RANK == 0:
                warnings.warn(f"'{cell_type_key}' column contains NaN values. scANVI will treat these as unlabeled cells.")

        try:
            SCANVI.setup_anndata(adata_scvi, layer="counts", batch_key=batch_key, labels_key=cell_type_key, unlabeled_category='Unknown')
            if LOCAL_RANK == 0:
                print("  AnnData setup for scANVI complete.")
        except Exception as e:
            raise RuntimeError(f"Error during scANVI AnnData setup: {e}. "
                               "Ensure 'counts' layer and batch_key are correctly set.")

        lvae = SCANVI(adata_scvi, n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers)
        if LOCAL_RANK == 0: # Guarded print
            print(f"  scANVI model initialized.")
            print(f"  Training scANVI for {max_epochs_scanvi} epochs (using {scanvi_trainer_kwargs['accelerator']} and {scanvi_trainer_kwargs['devices']} devices)...")
        lvae.train(
            max_epochs=max_epochs_scanvi,
            early_stopping=False, # Disable scvi-tools internal early stopping
            enable_progress_bar=True, # Keep True for scANVI
            train_size=0.9,
            validation_size=0.1,
            **scanvi_trainer_kwargs # Unpack scANVI specific trainer_kwargs here
        )
        if LOCAL_RANK == 0: # Guarded print
            scANVI_train_end_time = time.time()
            hours, rem = divmod(scANVI_train_end_time - scANVI_train_start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"  scANVI model training complete. Elapsed time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

        if LOCAL_RANK == 0: # Guarded print
            lvae.save(scanvi_model_full_path, overwrite=True)
            print(f"  scANVI model saved to {scanvi_model_full_path}")
    else:
        if LOCAL_RANK == 0:
            print(f"\nStep 3: Existing scANVI model found at {scanvi_model_full_path}. Skipping training and loading model...")
        lvae = SCANVI.load(scanvi_model_full_path, adata=adata_scvi)
        if LOCAL_RANK == 0:
            print("  scANVI model loaded successfully.")

    # Ensure model is on the correct device for inference (important if loaded from disk)
    if use_cuda and hasattr(lvae, 'module') and hasattr(lvae.module, 'to'):
        # If num_gpus_for_scanvi_training > 1, model is on LOCAL_RANK via DDP.
        # If num_gpus_for_scanvi_training == 1, it's on cuda:0 as a single model.
        # So send it to LOCAL_RANK's device
        lvae.module.to(torch.device(f"cuda:{LOCAL_RANK}")) 
    elif use_cuda and hasattr(lvae, 'to'):
        lvae.to(torch.device(f"cuda:{LOCAL_RANK}"))

    adata.obsm["X_scANVI"] = lvae.get_latent_representation(adata=adata_scvi)
    if LOCAL_RANK == 0: # Guarded print
        print(f"  scANVI latent representation (X_scANVI) stored in adata.obsm with shape {adata.obsm['X_scANVI'].shape}.")

    # Synchronize all processes after scANVI training/loading
    if dist.is_initialized():
        dist.barrier()


    # --- Conditional Inference Block ---
    if run_inference:
        # --- Initialize distributed environment for inference if not already ---
        # This occurs if num_gpus_for_training was 1 or 0, or if DDP was not cleanly inherited from training
        # This explicit initialization will now happen ONLY if dist is not already initialized,
        # otherwise we assume the existing DDP environment.
        if not dist.is_initialized() and num_gpus_for_training > 1: # Only try to initialize if DDP is actually needed for inference
            if LOCAL_RANK == 0:
                print("  Re-initializing torch.distributed process group for inference phase...")
            try:
                # Ensure environment variables are set for re-initialization
                os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
                os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355') # Use a consistent port
                os.environ['RANK'] = str(LOCAL_RANK)
                os.environ['WORLD_SIZE'] = str(num_gpus_for_training) 
                dist.init_process_group("nccl", rank=LOCAL_RANK, world_size=num_gpus_for_training)
                if LOCAL_RANK == 0:
                    print("  torch.distributed process group re-initialized successfully for inference.")
            except Exception as e:
                if LOCAL_RANK == 0:
                    warnings.warn(f"Failed to re-initialize torch.distributed process group for inference: {e}. "
                                  "Inference will proceed on a single device per process.")
                # Fallback to single-process inference if DDP setup fails
                # This will make current_world_size_ddp = 1
        
        current_world_size_ddp = 1
        current_rank_ddp = 0
        if dist.is_initialized(): # Check again if it's initialized after potential re-init
            current_world_size_ddp = dist.get_world_size()
            current_rank_ddp = dist.get_rank()
        else: # Still not initialized (e.g., if num_gpus_for_training was 1 initially, or re-init failed)
            if LOCAL_RANK == 0:
                print("  INFO: torch.distributed process group not initialized. Inference will run on a single device.")

        # --- Use chunked AND distributed normalized expression generation for SCVI ---
        scvi_inference_start_time = time.time() # Start timer for SCVI inference
        if LOCAL_RANK == 0:
            print("\n  Generating batch-corrected normalized expression from SCVI using chunking (distributed across GPUs)...")
        
        try:
            # Calculate global slice for this rank
            total_cells_to_process = adata_scvi.shape[0]
            cells_per_rank = (total_cells_to_process + current_world_size_ddp - 1) // current_world_size_ddp
            start_cell_idx = current_rank_ddp * cells_per_rank
            end_cell_idx = min((current_rank_ddp + 1) * cells_per_rank, total_cells_to_process)

            # Subset adata_scvi for this rank's processing
            rank_adata_scvi_subset = adata_scvi[start_cell_idx:end_cell_idx].copy()

            # --- Ordered Printing for Inference Start ---
            # Each rank waits for its turn to print its subset info
            if dist.is_initialized(): 
                for rank_idx in range(current_world_size_ddp):
                    if current_rank_ddp == rank_idx:
                        print(f"  Rank {current_rank_ddp + 1}/{current_world_size_ddp} processing cells {start_cell_idx} to {end_cell_idx-1} (total {rank_adata_scvi_subset.shape[0]} cells).")
                    dist.barrier() # Wait for all processes to finish this print statement
            else: # Not distributed, just print (occurs if DDP was not initialized)
                print(f"  Rank {current_rank_ddp + 1}/{current_world_size_ddp} processing cells {start_cell_idx} to {end_cell_idx-1} (total {rank_adata_scvi_subset.shape[0]} cells).")
            # --- Ordered Printing for Inference End ---


            # Get normalized expression for highly variable genes (HVGs) from the chunked utility
            normalized_expression_hvg_rank_array = get_normalized_expression_chunked(
                model=vae,
                adata=rank_adata_scvi_subset,
                n_samples=n_samples_normalize_expression,
                transform_batch=transform_batch_normalize_expression,
                chunk_size_cells=chunk_size_normalize_expression,
                verbose= (LOCAL_RANK == 0) # Only verbose for main process
            )

            # All ranks save their portion to a unique temporary file
            temp_dir = os.path.join(output_dir, "temp_inference_chunks")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, f"scvi_rank_{current_rank_ddp}_chunk.npy")
            np.save(temp_file_path, normalized_expression_hvg_rank_array)
            
            if LOCAL_RANK == 0:
                print(f"  Rank {current_rank_ddp + 1}/{current_world_size_ddp} saved its chunk to {temp_file_path}.")
            
            # Synchronize all processes to ensure all files are written
            if dist.is_initialized():
                dist.barrier()
            
            # Only rank 0 loads and combines the files
            if LOCAL_RANK == 0:
                print(f"  Rank {current_rank_ddp + 1}/{current_world_size_ddp}: Gathering inference results from all GPUs (via files to CPU)...")
                combined_chunks = []
                for i in range(current_world_size_ddp):
                    chunk_path = os.path.join(temp_dir, f"scvi_rank_{i}_chunk.npy")
                    combined_chunks.append(np.load(chunk_path))
                    os.remove(chunk_path) # Clean up temporary file
                
                combined_normalized_expression_hvg_array = np.vstack(combined_chunks)
                print(f"  Rank {current_rank_ddp + 1}/{current_world_size_ddp}: All inference chunks combined.")

                # Map back to full gene set of the original adata
                scvi_normalized_full_genes = np.zeros((adata.shape[0], adata.shape[1]), dtype=np.float32)
                genes_to_use_indices = adata.var_names.get_indexer(genes_to_use)
                scvi_normalized_full_genes[:, genes_to_use_indices] = combined_normalized_expression_hvg_array
                
                adata.layers["scvi_normalized"] = scvi_normalized_full_genes
                print(f"  SCVI normalized expression successfully generated with {n_samples_normalize_expression} samples. Stored in adata.layers with shape {adata.layers['scvi_normalized'].shape}.")
            
            # Remove temporary directory by rank 0 after all is done and combined
            if LOCAL_RANK == 0:
                try:
                    shutil.rmtree(temp_dir) # Use rmtree for robust removal
                    print(f"  Removed temporary directory: {temp_dir}")
                except OSError as e:
                    warnings.warn(f"Could not remove temporary directory {temp_dir}: {e}. It might contain other files.")

        except Exception as e:
            if LOCAL_RANK == 0:
                warnings.warn(f"Failed to generate SCVI normalized expression even with chunking and distribution: {e}. Skipping this layer.")
            # Do NOT add zero arrays if inference failed and run_inference is True, per user request.
            if dist.is_initialized(): # Only exit if DDP was attempted and initialized
                dist.barrier() # Ensure all processes reach this point before potential exit
                sys.exit(1)
        if LOCAL_RANK == 0:
            scvi_inference_end_time = time.time()
            hours, rem = divmod(scvi_inference_end_time - scvi_inference_start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"  SCVI inference complete. Elapsed time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


        # --- Use chunked AND distributed normalized expression generation for scANVI ---
        scANVI_inference_start_time = time.time() # Start timer for scANVI inference
        if LOCAL_RANK == 0:
            print("  Generating batch-corrected normalized expression from scANVI using chunking (distributed across GPUs)...")
        
        # Re-use current_world_size_ddp, current_rank_ddp (which are 1 and 0 if DDP wasn't initialized)
        try:
            total_cells_to_process = adata_scvi.shape[0]
            cells_per_rank = (total_cells_to_process + current_world_size_ddp - 1) // current_world_size_ddp
            start_cell_idx = current_rank_ddp * cells_per_rank
            end_cell_idx = min((current_rank_ddp + 1) * cells_per_rank, total_cells_to_process)

            rank_adata_scvi_subset = adata_scvi[start_cell_idx:end_cell_idx].copy()
            # Print for all GPUs, with ranks +1:
            print(f"  Rank {current_rank_ddp + 1}/{current_world_size_ddp} processing cells {start_cell_idx} to {end_cell_idx-1} (total {rank_adata_scvi_subset.shape[0]} cells).")

            normalized_expression_hvg_rank_array = get_normalized_expression_chunked(
                model=lvae,
                adata=rank_adata_scvi_subset,
                n_samples=n_samples_normalize_expression,
                transform_batch=transform_batch_normalize_expression,
                chunk_size_cells=chunk_size_normalize_expression,
                verbose= (LOCAL_RANK == 0)
            )
            
            # All ranks save their portion to a unique temporary file
            temp_dir = os.path.join(output_dir, "temp_inference_chunks")
            os.makedirs(temp_dir, exist_ok=True) # Ensure temp_dir exists for all ranks
            temp_file_path = os.path.join(temp_dir, f"scanvi_rank_{current_rank_ddp}_chunk.npy") # Unique filename
            np.save(temp_file_path, normalized_expression_hvg_rank_array)
            
            if LOCAL_RANK == 0:
                print(f"  Rank {current_rank_ddp + 1}/{current_world_size_ddp} saved its chunk to {temp_file_path}.")
            
            # Synchronize all processes to ensure all files are written
            if dist.is_initialized():
                dist.barrier()
            
            # Only rank 0 loads and combines the files
            if LOCAL_RANK == 0:
                print(f"  Rank {current_rank_ddp + 1}/{current_world_size_ddp}: Gathering inference results from all GPUs (via files to CPU)...")
                combined_chunks = []
                for i in range(current_world_size_ddp):
                    chunk_path = os.path.join(temp_dir, f"scanvi_rank_{i}_chunk.npy")
                    combined_chunks.append(np.load(chunk_path))
                    os.remove(chunk_path) # Clean up temporary file
                
                combined_normalized_expression_hvg_array = np.vstack(combined_chunks)
                print(f"  Rank {current_rank_ddp + 1}/{current_world_size_ddp}: All inference chunks combined.")

                # Map back to full gene set of the original adata
                scanvi_normalized_full_genes = np.zeros((adata.shape[0], adata.shape[1]), dtype=np.float32)
                genes_to_use_indices = adata.var_names.get_indexer(genes_to_use)
                scanvi_normalized_full_genes[:, genes_to_use_indices] = combined_normalized_expression_hvg_array
                
                adata.layers["scanvi_normalized"] = scanvi_normalized_full_genes
                print(f"  scANVI normalized expression successfully generated with {n_samples_normalize_expression} samples. Stored in adata.layers with shape {adata.layers['scanvi_normalized'].shape}.")
            
            # Remove temporary directory by rank 0 after all is done and combined
            if LOCAL_RANK == 0:
                try:
                    shutil.rmtree(temp_dir) # Use rmtree for robust removal
                    print(f"  Removed temporary directory: {temp_dir}")
                except OSError as e:
                    warnings.warn(f"Could not remove temporary directory {temp_dir}: {e}. It might contain other files.")


        except Exception as e:
            if LOCAL_RANK == 0:
                warnings.warn(f"Failed to generate scANVI normalized expression even with chunking and distribution: {e}. Skipping this layer.")
            # Do NOT add zero arrays if inference failed and run_inference is True, per user request.
            if dist.is_initialized(): # Only exit if DDP was attempted and initialized
                dist.barrier() # Ensure all processes reach this point before potential exit
                sys.exit(1)
        if LOCAL_RANK == 0:
            scANVI_inference_end_time = time.time()
            hours, rem = divmod(scANVI_inference_end_time - scANVI_inference_start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"  scANVI inference complete. Elapsed time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    # --- scANVI Checkpoint Save (using same filename as SCVI checkpoint) ---
    scvi_output_path_final = os.path.join(output_dir, output_filename)
    if LOCAL_RANK == 0: # Only save from the main process
        print(f"\nStep 3.5: Saving scANVI-integrated AnnData object to {scvi_output_path_final} (checkpoint after scANVI)...")
        try:
            adata.write_h5ad(scvi_output_path_final)
            print("  scANVI-integrated AnnData object saved successfully.")
        except Exception as e:
            warnings.warn(f"Failed to save scANVI-integrated AnnData object: {e}. This might indicate a permissions or disk space issue.")
    if dist.is_initialized(): # Synchronize all processes before continuing
        dist.barrier()


    umap_start_time = time.time() # Start timer for UMAP
    if LOCAL_RANK == 0:
        print("\nStep 4: Computing UMAP for scVI and scANVI latent spaces...")

    # UMAP for scVI latent space
    adata.obsm["X_umap_scvi"] = np.empty((adata.shape[0], 2), dtype=np.float32) # Pre-allocate
    if LOCAL_RANK == 0: # Only compute UMAP on the main process
        sc.pp.neighbors(adata, n_neighbors=n_neighbors_umap, use_rep="X_scVI", random_state=random_state, key_added='neighbors_scvi')
        sc.tl.umap(adata, min_dist=min_dist_umap, random_state=random_state, neighbors_key='neighbors_scvi')
        adata.obsm['X_umap_scvi'] = adata.obsm['X_umap'].copy() # Copy from default UMAP slot
        print("  UMAP for scVI latent space computed and stored in `adata.obsm['X_umap_scvi']`.")

    if plot_umaps and LOCAL_RANK == 0: # Only plot from main process
        print("  Generating plots for scVI UMAP...")
        plot_integration_umaps(
            adata=adata,
            umap_key_to_plot='X_umap_scvi',
            plot_color_keys=plot_color_keys,
            output_dir=output_dir,
            output_filename_prefix="integrated_scvi_umaps",
            figure_size_inches=figure_size_inches,
            age_log2_plot_settings=age_log2_plot_settings,
            plot_title_suffix="scVI",
            use_highly_variable_genes_param=use_highly_variable_genes, # Passed for suptitle
            max_epochs_scvi_param=max_epochs_scvi, # Passed for suptitle
            max_epochs_scanvi_param=max_epochs_scanvi # Passed for suptitle
        )


    # UMAP for scANVI latent space
    adata.obsm["X_umap_scanvi"] = np.empty((adata.shape[0], 2), dtype=np.float32) # Pre-allocate
    if LOCAL_RANK == 0: # Only compute UMAP on the main process
        sc.pp.neighbors(adata, n_neighbors=n_neighbors_umap, use_rep="X_scANVI", random_state=random_state, key_added='neighbors_scanvi')
        sc.tl.umap(adata, min_dist=min_dist_umap, random_state=random_state, neighbors_key='neighbors_scanvi')
        adata.obsm['X_umap_scanvi'] = adata.obsm['X_umap'].copy() # Copy from default UMAP slot
        print("  UMAP for scANVI latent space computed and stored in `adata.obsm['X_umap_scanvi']`.")
    
    if plot_umaps and LOCAL_RANK == 0: # Only plot from main process
        print("  Generating plots for scANVI UMAP...")
        plot_integration_umaps(
            adata=adata,
            umap_key_to_plot='X_umap_scanvi',
            plot_color_keys=plot_color_keys,
            output_dir=output_dir,
            output_filename_prefix="integrated_scanvi_umaps",
            figure_size_inches=figure_size_inches,
            age_log2_plot_settings=age_log2_plot_settings,
            plot_title_suffix="scANVI",
            use_highly_variable_genes_param=use_highly_variable_genes, # Passed for suptitle
            max_epochs_scvi_param=max_epochs_scvi, # Passed for suptitle
            max_epochs_scanvi_param=max_epochs_scanvi # Passed for suptitle
        )
    if LOCAL_RANK == 0:
        umap_end_time = time.time()
        hours, rem = divmod(umap_end_time - umap_start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"  UMAP computation and plotting complete. Elapsed time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


    if LOCAL_RANK == 0:
        print("\nscVI/scANVI pipeline complete.")
    return adata


if __name__ == "__main__":
    # Start the timer for total execution time
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Run scVI/scANVI integration on a pre-existing Harmony output .h5ad file."
    )
    parser.add_argument(
        'harmony_run_name',
        type=str,
        help="The run name from the previous Harmony integration. "
             "E.g., 'velmeshev100k-wang50k_pcs20'. "
             "This name is used to locate the input .h5ad and define output paths."
    )
    # Add new argparse arguments
    parser.add_argument(
        '--run_inference',
        action='store_true', # Default is False
        help="If set, normalized expression layers will be generated after training."
    )
    parser.add_argument(
        '--overwrite_scvi_model',
        action='store_true', # Default is False. If set, train SCVI regardless of existing model.
        help="If set, SCVI model will be trained and overwritten regardless of existing model."
    )
    parser.add_argument(
        '--overwrite_scanvi_model',
        action='store_true', # Default is False. If set, train scANVI regardless of existing model.
        help="If set, scANVI model will be trained and overwritten regardless of existing model."
    )


    # Existing args for integration function
    parser.add_argument('--n_latent', type=int, default=30, help="Dimensionality of the latent space.")
    parser.add_argument('--n_hidden', type=int, default=128, help="Number of nodes per hidden layer.")
    parser.add_argument('--n_layers', type=int, default=2, help="Number of hidden layers.")
    parser.add_argument('--max_epochs_scvi', type=int, default=400, help="Max epochs for SCVI training.")
    parser.add_argument('--max_epochs_scanvi', type=int, default=20, help="Max epochs for scANVI training.")
    parser.add_argument('--batch_key', type=str, default='chemistry', help="Observation key for batch.")
    parser.add_argument('--cell_type_key', type=str, default='lineage', help="Observation key for cell types (scANVI).")
    parser.add_argument('--n_samples_normalize_expression', type=int, default=10, help="Number of Monte Carlo samples for normalized expression.")
    parser.add_argument('--transform_batch_normalize_expression', type=str, default='multiome', help="Transform batch for normalized expression.")
    parser.add_argument('--plot_umaps', type=lambda x: (str(x).lower() == 'true'), default=True, help="Whether to plot UMAPs.") # Changed to handle boolean from string
    parser.add_argument('--use_highly_variable_genes', type=lambda x: (str(x).lower() == 'true'), default=True, help="Use highly variable genes.") # Changed to handle boolean from string
    parser.add_argument('--n_neighbors_umap', type=int, default=30, help="Number of neighbors for UMAP.")
    parser.add_argument('--min_dist_umap', type=float, default=0.3, help="Min distance for UMAP.")
    parser.add_argument('--num_workers_scvi', type=int, default=8, help="Number of DataLoader workers for scVI/scANVI.")
    parser.add_argument('--chunk_size_normalize_expression', type=int, default=None, help="Chunk size for normalized expression inference. If None, estimated automatically.")
    parser.add_argument('--num_gpus_for_training', type=int, default=4, help="Number of GPUs for SCVI training.")
    parser.add_argument('--num_gpus_for_scanvi_training', type=int, default=4, help="Number of GPUs for scANVI training (fixed to 4 for testing).") # Changed default here
    parser.add_argument('--figure_size_inches', type=int, nargs=2, default=[20, 10], help="Width and height of UMAP figures (e.g., --figure_size_inches 20 10).")
    parser.add_argument('--random_state', type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # --- Initialize torch.distributed for the entire script ---
    # This must happen before any DDP-related code runs.
    # WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT are typically set by the launcher (e.g., srun or torchrun)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1 and not dist.is_initialized():
        # Only initialize if DDP is intended (WORLD_SIZE > 1) and not already initialized
        # (e.g., if run directly via python vs. via torch.distributed.run)
        try:
            # os.environ values are typically strings
            rank = int(os.environ.get('RANK', '0'))
            world_size = int(os.environ.get('WORLD_SIZE', '1'))
            master_addr = os.environ.get('MASTER_ADDR', 'localhost')
            master_port = os.environ.get('MASTER_PORT', '12355') # Use consistent port

            os.environ['RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            if LOCAL_RANK == 0:
                print(f"INFO: torch.distributed process group initialized (Rank {rank}/{world_size}, Master: {master_addr}:{master_port}).")
        except Exception as e:
            if LOCAL_RANK == 0:
                warnings.warn(f"Failed to initialize torch.distributed process group globally: {e}. "
                              "Proceeding in single-process mode for the entire script.")
            # Ensure dist.is_initialized() remains False if init failed
    # --- End of global torch.distributed initialization ---


    output_base_dir = "snRNAseq/outputs"
    harmony_input_dir = os.path.join(output_base_dir, args.harmony_run_name)
    harmony_input_h5ad_path = os.path.join(harmony_input_dir, f"{args.harmony_run_name}.h5ad")

    scvi_output_specific_dir = harmony_input_dir
    scvi_output_h5ad_filename = f"{args.harmony_run_name}_scvi.h5ad"

    plot_keys_for_umap = ['origin', 'dataset', 'chemistry', 'age_log2', 'lineage', 'tissue']
    age_plot_settings = {'age_ticks': np.array([0, 1, 3, 5, 9, 25])}

    if LOCAL_RANK == 0: # Only load and print summary from rank 0
        print(f"Attempting to load Harmony integrated data from: {harmony_input_h5ad_path}")
    try:
        integrated_adata_harmony = sc.read_h5ad(harmony_input_h5ad_path)
        if LOCAL_RANK == 0:
            print("AnnData object loaded successfully.")
            print("--- Loaded AnnData summary ---")
            print(integrated_adata_harmony)
            print("Layers:", integrated_adata_harmony.layers.keys())
            print("obsm:", integrated_adata_harmony.obsm.keys())
            print("------------------------------")

        if LOCAL_RANK == 0:
            print(f"\nStarting scVI/scANVI integration pipeline for Harmony run: {args.harmony_run_name}")
        
        integrated_adata_scvi = integrate_snrnaseq_scvi(
            adata=integrated_adata_harmony,
            output_dir=scvi_output_specific_dir,
            output_filename=scvi_output_h5ad_filename,
            scvi_model_path="scvi_model",
            scanvi_model_path="scanvi_model",
            n_latent=args.n_latent,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            max_epochs_scvi=args.max_epochs_scvi,
            max_epochs_scanvi=args.max_epochs_scanvi,
            batch_key=args.batch_key,
            cell_type_key=args.cell_type_key,
            n_samples_normalize_expression=args.n_samples_normalize_expression,
            transform_batch_normalize_expression=args.transform_batch_normalize_expression,
            plot_umaps=args.plot_umaps,
            plot_color_keys=plot_keys_for_umap,
            figure_size_inches=tuple(args.figure_size_inches), # Convert list from argparse to tuple
            age_log2_plot_settings=age_plot_settings,
            random_state=args.random_state,
            use_highly_variable_genes=args.use_highly_variable_genes,
            n_neighbors_umap=args.n_neighbors_umap,
            min_dist_umap=args.min_dist_umap,
            num_workers_scvi=args.num_workers_scvi,
            chunk_size_normalize_expression=args.chunk_size_normalize_expression,
            num_gpus_for_training=args.num_gpus_for_training,
            num_gpus_for_scanvi_training=args.num_gpus_for_scanvi_training,
            overwrite_scvi_model=args.overwrite_scvi_model,
            overwrite_scanvi_model=args.overwrite_scanvi_model,
            run_inference=args.run_inference,
        )

        if LOCAL_RANK == 0:
            print(f"\nscVI/scANVI integration successfully completed for {args.harmony_run_name}.")
            # The final save message here refers to the last checkpoint, not a new save.
            print(f"Final AnnData object (saved at checkpoint) structure after scVI/scANVI:")
            print(integrated_adata_scvi)
            print("Latent representations in .obsm:", integrated_adata_scvi.obsm.keys())
            print("Layers in .layers:", integrated_adata_scvi.layers.keys())

    except FileNotFoundError as e:
        if LOCAL_RANK == 0:
            print(f"ERROR: Input .h5ad file not found at {harmony_input_h5ad_path}. "
                  "Please ensure the 'harmony_run_name' is correct and the file exists. Details: {e}")
        sys.exit(1)
    except ValueError as e:
        if LOCAL_RANK == 0:
            print(f"ERROR: Configuration or data value issue. Details: {e}")
        sys.exit(1)
    except RuntimeError as e:
        if LOCAL_RANK == 0:
            print(f"ERROR: A critical processing step failed. Details: {e}")
        sys.exit(1)
    except Exception as e:
        if LOCAL_RANK == 0:
            print(f"An unexpected ERROR occurred during pipeline execution: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up DDP process group if initialized
        if dist.is_initialized():
            dist.destroy_process_group()
            if LOCAL_RANK == 0:
                print("INFO: torch.distributed process group destroyed.")
        
        if LOCAL_RANK == 0: # Only print total time from rank 0
            # Calculate and print total execution time
            end_time = time.time()
            total_time_seconds = end_time - start_time
            hours, rem = divmod(total_time_seconds, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"\nTotal script execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
