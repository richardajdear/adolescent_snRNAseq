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

# Import the new plotting utility function
sys.path.append(os.path.join(os.path.dirname(__file__))) # Add current directory to path
try:
    from plot_utils import plot_integration_umaps
except ImportError:
    print("Error: Could not import plot_integration_umaps from plot_utils.py.")
    print("Please ensure 'plot_utils.py' is in the 'snRNAseq/code' directory.")
    sys.exit(1)

try:
    import scvi
    from scvi.model import SCVI, SCANVI
except ImportError:
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
    n_samples_normalize_expression: int = 1,
    transform_batch_normalize_expression: str = 'multiome',
    plot_umaps: bool = True,
    plot_color_keys: list[str] = None,
    figure_size_inches: tuple[int, int] = (20, 10),
    age_log2_plot_settings: dict = None,
    random_state: int = 42,
    use_highly_variable_genes: bool = True,
    n_neighbors_umap: int = 30,
    min_dist_umap: float = 0.3,
    num_workers_scvi: int = 8, # New parameter for scVI/scANVI DataLoader workers
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
        n_samples_normalize_expression: Number of samples to draw for `get_normalized_expression`.
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

    print("\n--- scVI/scANVI Integration Pipeline ---")

    scvi.settings.seed = random_state
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_device(0)
        # Apply Tensor Core optimization
        torch.set_float32_matmul_precision('high')
        print("  Set torch.set_float32_matmul_precision('high') for Tensor Core optimization.")
    else:
        print("CUDA not available. Using CPU.")

    # Set scvi-tools number of workers
    scvi.settings.num_workers = num_workers_scvi
    print(f"  scvi.settings.num_workers set to {num_workers_scvi}.")


    # Save the original X_umap (from Harmony) to X_umap_harmony
    if 'X_umap' in adata.obsm:
        adata.obsm['X_umap_harmony'] = adata.obsm['X_umap'].copy()
        print("  Original 'X_umap' (from Harmony) copied to 'X_umap_harmony' for preservation.")
    else:
        warnings.warn("No 'X_umap' found in the input AnnData object. "
                      "Harmony UMAP cannot be saved to 'X_umap_harmony'. "
                      "Ensure the input AnnData is from a Harmony integration that computed UMAP.")

    print("\nStep 1: Preparing AnnData for scVI...")
    if 'counts' not in adata.layers:
        print("  Storing .X as 'counts' layer for scVI.")
        adata.layers["counts"] = adata.X.copy()
    else:
        print("  'counts' layer already exists. Using existing raw counts.")

    if use_highly_variable_genes:
        if 'highly_variable' not in adata.var.columns:
            warnings.warn("`use_highly_variable_genes` is True, but 'highly_variable' not found in .var. "
                          "Training will proceed on all genes. Please ensure highly variable genes are identified "
                          "or set `use_highly_variable_genes=False`.")
            genes_to_use = adata.var_names
            print("  Training on all genes due to missing 'highly_variable' flag.")
        else:
            genes_to_use = adata.var_names[adata.var['highly_variable']]
            print(f"  Using {len(genes_to_use)} highly variable genes for training.")
    else:
        genes_to_use = adata.var_names
        print("  Training on all genes as `use_highly_variable_genes` is False.")

    adata_scvi = adata[:, genes_to_use].copy()
    adata_scvi.X = adata_scvi.layers['counts'].copy()

    print("  Setting up AnnData for scVI...")
    try:
        SCVI.setup_anndata(adata_scvi, layer="counts", batch_key=batch_key)
        print("  AnnData setup for scVI complete.")
    except Exception as e:
        raise RuntimeError(f"Error during scVI AnnData setup: {e}. "
                           "Ensure 'counts' layer and batch_key are correctly set.")

    print("\nStep 2: Training SCVI model...")
    vae = SCVI(adata_scvi, n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers)
    print(f"  SCVI model initialized with n_latent={n_latent}, n_hidden={n_hidden}, n_layers={n_layers}.")
    print(f"  Training SCVI for {max_epochs_scvi} epochs...")
    vae.train(max_epochs=max_epochs_scvi, early_stopping=True, enable_progress_bar=True, train_size=0.9, validation_size=0.1)

    scvi_model_full_path = os.path.join(output_dir, scvi_model_path)
    vae.save(scvi_model_full_path, overwrite=True)
    print(f"  SCVI model saved to {scvi_model_full_path}")

    adata.obsm["X_scVI"] = vae.get_latent_representation(adata=adata_scvi)
    print(f"  SCVI latent representation (X_scVI) stored in adata.obsm with shape {adata.obsm['X_scVI'].shape}.")

    print("  Generating batch-corrected normalized expression from SCVI...")
    scvi_normalized_expression = vae.get_normalized_expression(
        adata=adata_scvi,
        n_samples=n_samples_normalize_expression,
        transform_batch=transform_batch_normalize_expression
    )
    scvi_normalized_full = np.zeros((adata.shape[0], adata.shape[1]))
    genes_to_use_indices = adata.var_names.get_indexer(genes_to_use)
    scvi_normalized_full[:, genes_to_use_indices] = scvi_normalized_expression.values
    adata.layers["scvi_normalized"] = scvi_normalized_full
    print(f"  SCVI normalized and batch-corrected expression (scvi_normalized) stored in adata.layers with shape {adata.layers['scvi_normalized'].shape}.")


    print(f"\nStep 3: Training scANVI model using '{cell_type_key}' as cell type key...")
    if cell_type_key not in adata_scvi.obs.columns:
        raise ValueError(f"Cell type key '{cell_type_key}' not found in adata_scvi.obs. "
                           "scANVI requires a cell type annotation for semi-supervised training.")
    if adata_scvi.obs[cell_type_key].isnull().any():
        warnings.warn(f"'{cell_type_key}' column contains NaN values. scANVI will treat these as unlabeled cells.")

    try:
        SCANVI.setup_anndata(adata_scvi, layer="counts", batch_key=batch_key, labels_key=cell_type_key, unlabeled_category='Unknown')
        print("  AnnData setup for scANVI complete.")
    except Exception as e:
        raise RuntimeError(f"Error during scANVI AnnData setup: {e}. "
                           "Ensure 'counts' layer, batch_key, and labels_key are correctly set.")

    lvae = SCANVI(adata_scvi, n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers)
    print(f"  scANVI model initialized.")
    lvae.vae_model = vae
    print(f"  Training scANVI for {max_epochs_scanvi} epochs...")
    lvae.train(max_epochs=max_epochs_scanvi, early_stopping=True, enable_progress_bar=True, train_size=0.9, validation_size=0.1)

    scanvi_model_full_path = os.path.join(output_dir, scanvi_model_path)
    lvae.save(scanvi_model_full_path, overwrite=True)
    print(f"  scANVI model saved to {scanvi_model_full_path}")

    adata.obsm["X_scANVI"] = lvae.get_latent_representation(adata=adata_scvi)
    print(f"  scANVI latent representation (X_scANVI) stored in adata.obsm with shape {adata.obsm['X_scANVI'].shape}.")

    print("  Generating batch-corrected normalized expression from scANVI...")
    scanvi_normalized_expression = lvae.get_normalized_expression(
        adata=adata_scvi,
        n_samples=n_samples_normalize_expression,
        transform_batch=transform_batch_normalize_expression
    )
    scanvi_normalized_full = np.zeros((adata.shape[0], adata.shape[1]))
    scanvi_normalized_full[:, genes_to_use_indices] = scanvi_normalized_expression.values
    adata.layers["scanvi_normalized"] = scanvi_normalized_full
    print(f"  scANVI normalized and batch-corrected expression (scanvi_normalized) stored in adata.layers with shape {adata.layers['scanvi_normalized'].shape}.")


    print("\nStep 4: Computing UMAP for scVI and scANVI latent spaces...")

    # UMAP for scVI latent space
    sc.pp.neighbors(adata, n_neighbors=n_neighbors_umap, use_rep="X_scVI", random_state=random_state, key_added='neighbors_scvi')
    sc.tl.umap(adata, min_dist=min_dist_umap, random_state=random_state, neighbors_key='neighbors_scvi')
    adata.obsm['X_umap_scvi'] = adata.obsm['X_umap'].copy()
    print("  UMAP for scVI latent space computed and stored in `adata.obsm['X_umap_scvi']`.")
    if plot_umaps:
        print("  Generating plots for scVI UMAP...")
        plot_integration_umaps(
            adata=adata,
            umap_key_to_plot='X_umap_scvi',
            plot_color_keys=plot_color_keys,
            output_dir=output_dir,
            output_filename_prefix="integrated_scvi_umaps",
            figure_size_inches=figure_size_inches,
            age_log2_plot_settings=age_log2_plot_settings,
            plot_title_suffix="scVI", # Keep this for subplot titles
            add_suptitle=False # Explicitly add suptitle if desired, otherwise it's off by default in the plot_utils function
        )


    # UMAP for scANVI latent space
    sc.pp.neighbors(adata, n_neighbors=n_neighbors_umap, use_rep="X_scANVI", random_state=random_state, key_added='neighbors_scanvi')
    sc.tl.umap(adata, min_dist=min_dist_umap, random_state=random_state, neighbors_key='neighbors_scanvi')
    adata.obsm['X_umap_scanvi'] = adata.obsm['X_umap'].copy()
    print("  UMAP for scANVI latent space computed and stored in `adata.obsm['X_umap_scanvi']`.")
    if plot_umaps:
        print("  Generating plots for scANVI UMAP...")
        plot_integration_umaps(
            adata=adata,
            umap_key_to_plot='X_umap_scanvi',
            plot_color_keys=plot_color_keys,
            output_dir=output_dir,
            output_filename_prefix="integrated_scanvi_umaps",
            figure_size_inches=figure_size_inches,
            age_log2_plot_settings=age_log2_plot_settings,
            plot_title_suffix="scANVI", # Keep this for subplot titles
            add_suptitle=False # Explicitly add suptitle if desired, otherwise it's off by default in the plot_utils function
        )


    # Save Processed Data
    print(f"\nStep 5: Saving integrated AnnData object to {output_dir}/{output_filename}...")
    output_path = os.path.join(output_dir, output_filename)
    try:
        adata.write_h5ad(output_path)
        print("  AnnData object saved successfully.")
    except Exception as e:
        warnings.warn(f"Failed to save AnnData object: {e}. This might indicate a permissions or disk space issue.")

    print("\nscVI/scANVI pipeline complete.")
    return adata


if __name__ == "__main__":
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
    args = parser.parse_args()

    output_base_dir = "snRNAseq/outputs"
    harmony_input_dir = os.path.join(output_base_dir, args.harmony_run_name)
    harmony_input_h5ad_path = os.path.join(harmony_input_dir, f"{args.harmony_run_name}.h5ad")

    scvi_output_specific_dir = harmony_input_dir
    scvi_output_h5ad_filename = f"{args.harmony_run_name}_scvi.h5ad"

    plot_keys_for_umap = ['origin', 'dataset', 'chemistry', 'age_log2', 'lineage', 'tissue']
    age_plot_settings = {'age_ticks': np.array([0, 1, 3, 5, 9, 25])}

    print(f"Attempting to load Harmony integrated data from: {harmony_input_h5ad_path}")
    try:
        integrated_adata_harmony = sc.read_h5ad(harmony_input_h5ad_path)
        print("AnnData object loaded successfully.")
        print("--- Loaded AnnData summary ---")
        print(integrated_adata_harmony)
        print("Layers:", integrated_adata_harmony.layers.keys())
        print("obsm:", integrated_adata_harmony.obsm.keys())
        print("------------------------------")

        print(f"\nStarting scVI/scANVI integration pipeline for Harmony run: {args.harmony_run_name}")
        integrated_adata_scvi = integrate_snrnaseq_scvi(
            adata=integrated_adata_harmony,
            output_dir=scvi_output_specific_dir,
            output_filename=scvi_output_h5ad_filename,
            scvi_model_path="scvi_model",
            scanvi_model_path="scanvi_model",
            n_latent=30,
            n_hidden=128,
            n_layers=2,
            max_epochs_scvi=400,
            max_epochs_scanvi=20,
            batch_key='chemistry',
            cell_type_key='lineage',
            n_samples_normalize_expression=1,
            transform_batch_normalize_expression='multiome',
            plot_umaps=True,
            plot_color_keys=plot_keys_for_umap,
            figure_size_inches=(20, 10),
            age_log2_plot_settings=age_plot_settings,
            random_state=42,
            use_highly_variable_genes=False,
            n_neighbors_umap=30,
            min_dist_umap=0.3,
            num_workers_scvi=min(os.cpu_count() or 1, 8), # Default to 8 or available CPUs, whichever is lower
        )

        print(f"\nscVI/scANVI integration successfully completed for {args.harmony_run_name}.")
        print(f"Integrated AnnData object saved to: {os.path.join(scvi_output_specific_dir, scvi_output_h5ad_filename)}")
        print("Final AnnData object structure after scVI/scANVI:")
        print(integrated_adata_scvi)
        print("Latent representations in .obsm:", integrated_adata_scvi.obsm.keys())
        print("Layers in .layers:", integrated_adata_scvi.layers.keys())

    except FileNotFoundError as e:
        print(f"ERROR: Input .h5ad file not found at {harmony_input_h5ad_path}. "
              "Please ensure the 'harmony_run_name' is correct and the file exists. Details: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: Configuration or data value issue. Details: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"ERROR: A critical processing step failed. Details: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected ERROR occurred during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
