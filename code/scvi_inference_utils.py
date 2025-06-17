import anndata as ad
import numpy as np
import torch
import warnings
from typing import Optional

def get_normalized_expression_chunked(
    model,
    adata: ad.AnnData,
    n_samples: int = 1,
    transform_batch: Optional[str] = None,
    chunk_size_cells: Optional[int] = None,
    target_vram_fraction: float = 0.25, # Fraction of GPU VRAM to target per chunk (e.g., 0.25 for 1/4th of 80GB)
    verbose: bool = True,
) -> np.ndarray:
    """
    Generates normalized expression by processing cells in chunks to manage memory.

    This function is a wrapper around scvi.model.SCVI.get_normalized_expression,
    designed to handle large datasets by splitting the AnnData object into smaller
    chunks, performing inference on each chunk, and then recombining the results.

    Args:
        model: A trained scvi.model.SCVI or scvi.model.SCANVI model.
        adata: The AnnData object (or a subset of it, e.g., adata_scvi) containing
               the data the model was trained on.
        n_samples: Number of Monte Carlo samples to draw for each cell.
        transform_batch: Batch to transform the data to. Passed to model.get_normalized_expression.
        chunk_size_cells: Optional. The number of cells to process in each chunk.
                          If None, an appropriate chunk size will be estimated based
                          on available GPU VRAM and data dimensions.
        target_vram_fraction: If chunk_size_cells is None, this fraction of a single
                              GPU's VRAM will be targeted for the intermediate
                              (n_samples * chunk_size_cells * n_genes) tensor.
        verbose: If True, print detailed messages about chunking and progress.

    Returns:
        A NumPy array of shape (n_cells, n_genes) with normalized and batch-corrected
        (if transform_batch is used) expression values. This array is on CPU.
    """
    if verbose:
        print(f"  Attempting to generate normalized expression with {n_samples} samples per cell.")

    if not isinstance(adata, ad.AnnData):
        raise TypeError("Input 'adata' must be an AnnData object.")

    # Determine device for inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"  Inference will use device: {device}")

    n_cells = adata.shape[0]
    n_genes = adata.shape[1]

    if chunk_size_cells is None:
        # Estimate chunk size based on GPU VRAM
        bytes_per_float = 4 # float32
        # Assuming model.get_normalized_expression creates an intermediate tensor of shape (n_samples, n_cells_in_chunk, n_genes)
        # and then averages over n_samples to return (n_cells_in_chunk, n_genes).
        # We need to ensure the (n_samples, n_cells_in_chunk, n_genes) intermediate tensor fits in VRAM.

        # Get total VRAM on the current device
        total_gpu_vram_bytes = 0
        if device == 'cuda':
            try:
                # Use torch.cuda.get_device_properties to get VRAM for the current device
                # The model itself also consumes VRAM.
                total_gpu_vram_bytes = torch.cuda.get_device_properties(model.module.device).total_memory
                if verbose:
                    print(f"  Total VRAM on current GPU ({model.module.device}): {total_gpu_vram_bytes / (1024**3):.2f} GiB.")
            except Exception as e:
                warnings.warn(f"Could not get GPU VRAM properties: {e}. Falling back to default estimation.")
                # Fallback to a reasonable default if properties can't be fetched (e.g., 80GB for A100)
                total_gpu_vram_bytes = 80 * (1024**3) # Assume 80GB if cannot detect

        # If running on CPU, we target a fraction of system RAM.
        # But generally, for this error, it's VRAM. Let's assume VRAM target primarily.
        if total_gpu_vram_bytes == 0: # If still 0, might be CPU or failed VRAM detection
             # Default to a safe VRAM equivalent target if no GPU VRAM detected (e.g., 20 GiB for intermediate calculation)
            target_vram_bytes_for_chunk = 20 * (1024**3)
        else:
            target_vram_bytes_for_chunk = total_gpu_vram_bytes * target_vram_fraction

        # Memory needed for the intermediate (n_samples, chunk_cells, n_genes) tensor
        # This is where the OOM error occurs.
        estimated_memory_per_cell_all_samples_bytes = n_samples * n_genes * bytes_per_float

        if estimated_memory_per_cell_all_samples_bytes == 0:
            estimated_cells_per_chunk = n_cells # Avoid division by zero, process all at once if no genes/samples
        else:
            estimated_cells_per_chunk = target_vram_bytes_for_chunk / estimated_memory_per_cell_all_samples_bytes

        chunk_size_cells = max(1, int(estimated_cells_per_chunk)) # Ensure at least 1 cell per chunk
        # Set a reasonable upper bound for chunk_size_cells to prevent excessive memory usage even if calculation is too generous
        chunk_size_cells = min(chunk_size_cells, 50000) # Cap at, e.g., 50,000 cells per chunk to avoid extreme large chunks

        if verbose:
            print(f"  Estimated chunk size: {chunk_size_cells} cells (targeting {target_vram_fraction*100:.1f}% VRAM).")

    if chunk_size_cells >= n_cells:
        if verbose:
            print(f"  Chunk size ({chunk_size_cells}) is larger than or equal to total cells ({n_cells}). Processing all cells at once.")
        # If chunk size is larger than total cells, just process all at once
        normalized_expression_df = model.get_normalized_expression(
            adata=adata,
            n_samples=n_samples,
            transform_batch=transform_batch
        )
        return normalized_expression_df.values # Return NumPy array directly

    all_normalized_expression_chunks = []
    num_chunks = (n_cells + chunk_size_cells - 1) // chunk_size_cells

    if verbose:
        print(f"  Splitting data into {num_chunks} chunks of ~{chunk_size_cells} cells each.")

    for i in range(num_chunks):
        start_idx = i * chunk_size_cells
        end_idx = min((i + 1) * chunk_size_cells, n_cells)
        chunk_adata = adata[start_idx:end_idx].copy()

        if verbose:
            print(f"  Processing chunk {i+1}/{num_chunks} ({start_idx}-{end_idx-1} cells).")

        try:
            # Important: Ensure chunk_adata is on the correct device if needed by the model internally.
            # get_normalized_expression typically handles device transfer internally based on model device.
            normalized_expression_chunk_df = model.get_normalized_expression(
                adata=chunk_adata,
                n_samples=n_samples,
                transform_batch=transform_batch
            )
            all_normalized_expression_chunks.append(normalized_expression_chunk_df.values)
        except Exception as e:
            warnings.warn(f"Error processing chunk {i+1}: {e}. Skipping this chunk. "
                          "This may lead to missing data in the final normalized expression.")
            # If a chunk fails, append an array of zeros for that chunk to maintain shape
            all_normalized_expression_chunks.append(np.zeros((chunk_adata.shape[0], n_genes), dtype=np.float32))

    if not all_normalized_expression_chunks:
        warnings.warn("No normalized expression chunks were successfully processed. Returning empty array.")
        return np.zeros((n_cells, n_genes), dtype=np.float32)

    return np.vstack(all_normalized_expression_chunks)

