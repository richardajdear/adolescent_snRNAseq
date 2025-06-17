import scanpy as sc
import anndata as ad
import numpy as np
import os
import sys
import warnings
import torch
import argparse
from typing import Optional

# Add the directory containing scvi_inference_utils.py to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import the inference utility
try:
    from scvi_inference_utils import get_normalized_expression_chunked
except ImportError:
    print("Error: Could not import get_normalized_expression_chunked from scvi_inference_utils.py.")
    print("Please ensure 'scvi_inference_utils.py' is in the same directory as this script or in PYTHONPATH.")
    sys.exit(1)

# Import scvi and model classes
try:
    import scvi
    from scvi.model import SCVI, SCANVI
except ImportError:
    print("Error: Could not import scvi. Please ensure it is installed (`pip install scvi-tools`).")
    sys.exit(1)

def run_inference_and_normalize(
    adata_path: str,
    model_path: str,
    output_h5ad_path: str,
    n_samples_normalize_expression: int = 1,
    transform_batch_normalize_expression: Optional[str] = 'multiome',
    chunk_size_normalize_expression: Optional[int] = None,
    model_type: str = 'scvi', # 'scvi' or 'scanvi'
    target_layer_name: str = 'scvi_normalized_inference', # Name for the new layer
    cell_type_key: str = 'lineage', # Only relevant if model_type is 'scanvi'
    verbose: bool = True,
) -> ad.AnnData:
    """
    Loads a pre-trained scVI/scANVI model and an AnnData object,
    then generates and saves normalized expression data using chunking.

    Args:
        adata_path: Path to the input AnnData .h5ad file.
        model_path: Path to the trained model's saved state (e.g., 'path/to/model/model.pt').
        output_h5ad_path: Path to save the output AnnData .h5ad file with the new layer.
        n_samples_normalize_expression: Number of Monte Carlo samples to draw for `get_normalized_expression`.
        transform_batch_normalize_expression: Batch to transform the data to.
        chunk_size_normalize_expression: Optional. Number of cells to process in each chunk.
                                         If None, it will be estimated based on GPU VRAM.
        model_type: The type of model ('scvi' or 'scanvi').
        target_layer_name: The name of the new layer in adata.layers to store the normalized data.
        cell_type_key: The observation key for cell types, relevant for scANVI model loading setup.
        verbose: If True, print detailed messages.

    Returns:
        The AnnData object with the new normalized expression layer.
    """
    print("--- Starting scVI/scANVI Inference and Normalization ---")
    if verbose:
        print(f"Input AnnData path: {adata_path}")
        print(f"Model path: {model_path}")
        print(f"Output AnnData path: {output_h5ad_path}")
        print(f"Model type: {model_type}")

    # Set up CUDA if available, without setting device directly for model loading
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        if verbose:
            print(f"CUDA is available. Active device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
            torch.set_float32_matmul_precision('high')
            print("  Set torch.set_float32_matmul_precision('high') for Tensor Core optimization.")
    else:
        if verbose:
            print("CUDA not available. Using CPU.")

    # 1. Load AnnData object
    try:
        adata = sc.read_h5ad(adata_path)
        if verbose:
            print(f"AnnData object loaded successfully with shape: {adata.shape}")
            print(f"AnnData layers: {adata.layers.keys()}")
            print(f"AnnData obsm: {adata.obsm.keys()}")
    except Exception as e:
        raise RuntimeError(f"Failed to load AnnData from {adata_path}: {e}")

    # Ensure 'counts' layer exists and adata.X is set up for model loading
    if 'counts' not in adata.layers:
        warnings.warn("No 'counts' layer found in AnnData. Assuming .X contains raw counts and copying it.")
        adata.layers['counts'] = adata.X.copy()
    else:
        if verbose:
            print("Using existing 'counts' layer for model setup.")

    # Ensure highly variable genes are set up, or mark all if not
    if 'highly_variable' not in adata.var.columns:
        warnings.warn("No 'highly_variable' genes found in AnnData. Marking all genes as highly_variable for model loading setup.")
        adata.var['highly_variable'] = True
    
    # Get the subset of data the model was trained on
    genes_to_use = adata.var_names[adata.var['highly_variable']]
    adata_model_setup = adata[:, genes_to_use].copy()
    adata_model_setup.X = adata_model_setup.layers['counts'].copy() # Ensure .X is raw counts for setup

    # Setup anndata for scvi-tools using existing setup or re-doing it
    if '_scvi_setup_dict' not in adata_model_setup.uns:
        if model_type == 'scvi':
            SCVI.setup_anndata(adata_model_setup, layer="counts", batch_key='chemistry') # Assuming 'chemistry' as batch key
        elif model_type == 'scanvi':
            if cell_type_key not in adata_model_setup.obs.columns:
                raise ValueError(f"Cell type key '{cell_type_key}' not found in adata.obs for scANVI model loading.")
            SCANVI.setup_anndata(adata_model_setup, layer="counts", batch_key='chemistry', labels_key=cell_type_key, unlabeled_category='Unknown')
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'scvi' or 'scanvi'.")
        if verbose:
            print(f"scvi-tools setup completed for model loading (using batch_key='chemistry').")
    else:
        if verbose:
            print("scvi-tools setup dictionary already present. Proceeding with existing setup.")


    # 2. Load pre-trained model
    try:
        if model_type == 'scvi':
            model = SCVI.load(model_path, adata=adata_model_setup)
        elif model_type == 'scanvi':
            model = SCANVI.load(model_path, adata=adata_model_setup)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'scvi' or 'scanvi'.")
        if verbose:
            print(f"Model ({model_type}) loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    # Ensure model is on the correct device for inference
    if use_cuda and hasattr(model, 'module') and hasattr(model.module, 'to'):
        model.module.to(torch.device('cuda'))
        if verbose:
            print(f"Model moved to GPU for inference.")
    elif use_cuda and hasattr(model, 'to'): # For some older scvi-tools versions
        model.to(torch.device('cuda'))
        if verbose:
            print(f"Model moved to GPU for inference.")
    else:
        if verbose:
            print("Model will run on CPU for inference.")

    # 3. Generate normalized expression using chunking
    if verbose:
        print(f"Generating normalized expression data (n_samples={n_samples_normalize_expression})...")
    
    try:
        normalized_expression_array = get_normalized_expression_chunked(
            model=model,
            adata=adata_model_setup, # Use the subsetted adata for chunking
            n_samples=n_samples_normalize_expression,
            transform_batch=transform_batch_normalize_expression,
            chunk_size_cells=chunk_size_normalize_expression,
            verbose=verbose
        )
        if verbose:
            print(f"Normalized expression data generated with shape: {normalized_expression_array.shape}")
        
        # Map the normalized expression back to the full gene set of the original adata
        final_normalized_full = np.zeros((adata.shape[0], adata.shape[1]), dtype=np.float32)
        original_hvg_indices = adata.var_names.get_indexer(genes_to_use)
        final_normalized_full[:, original_hvg_indices] = normalized_expression_array
        
        adata.layers[target_layer_name] = final_normalized_full
        if verbose:
            print(f"Normalized expression saved to adata.layers['{target_layer_name}'] with shape {adata.layers[target_layer_name].shape}.")

    except Exception as e:
        warnings.warn(f"Failed to generate normalized expression even with chunking: {e}. Skipping this layer.")
        adata.layers[target_layer_name] = np.zeros((adata.shape[0], adata.shape[1]), dtype=np.float32) # Placeholder


    # 4. Save the modified AnnData object
    try:
        adata.write_h5ad(output_h5ad_path)
        if verbose:
            print(f"Modified AnnData object saved successfully to {output_h5ad_path}.")
    except Exception as e:
        raise RuntimeError(f"Failed to save modified AnnData object to {output_h5ad_path}: {e}")

    print("--- Inference and Normalization pipeline complete. ---")
    return adata

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Run scVI/scANVI inference to generate normalized expression on pre-trained model and data."
    )
    parser.add_argument(
        '--adata_path',
        type=str,
        required=True,
        help="Full path to the input AnnData .h5ad file (e.g., /path/to/data.h5ad)."
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Full path to the trained model's saved state (e.g., /path/to/model/model.pt)."
    )
    parser.add_argument(
        '--output_suffix',
        type=str,
        default='_inference_normalized',
        help="Suffix for the output .h5ad filename. Output will be saved in the same directory as input adata."
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=10,
        help="Number of Monte Carlo samples for normalized expression."
    )
    parser.add_argument(
        '--transform_batch',
        type=str,
        default='multiome',
        help="Batch to transform the data to ('multiome' or a specific batch value)."
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=None,
        help="Optional. Number of cells per chunk for normalization. If None, estimated automatically."
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='scvi',
        choices=['scvi', 'scanvi'],
        help="Type of the trained model (scvi or scanvi)."
    )
    parser.add_argument(
        '--target_layer_name',
        type=str,
        default='scvi_normalized_inference',
        help="Name of the new layer to save normalized data in AnnData."
    )
    parser.add_argument(
        '--cell_type_key',
        type=str,
        default='lineage',
        help="Cell type key in .obs, required if model_type is 'scanvi'."
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose output."
    )

    args = parser.parse_args()

    # Determine output path dynamically
    adata_dir = os.path.dirname(args.adata_path)
    adata_filename_base = os.path.splitext(os.path.basename(args.adata_path))[0]
    output_h5ad_path = os.path.join(adata_dir, f"{adata_filename_base}{args.output_suffix}.h5ad")

    try:
        run_inference_and_normalize(
            adata_path=args.adata_path,
            model_path=args.model_path,
            output_h5ad_path=output_h5ad_path,
            n_samples_normalize_expression=args.n_samples,
            transform_batch_normalize_expression=args.transform_batch,
            chunk_size_normalize_expression=args.chunk_size,
            model_type=args.model_type,
            target_layer_name=args.target_layer_name,
            cell_type_key=args.cell_type_key,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"An ERROR occurred during inference pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        end_time = time.time()
        total_time_seconds = end_time - start_time
        hours, rem = divmod(total_time_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nTotal script execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
