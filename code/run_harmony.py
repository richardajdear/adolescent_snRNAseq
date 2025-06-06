import scanpy as sc
import scanpy.external as sce
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings # Import warnings for more controlled error reporting

# Assume 'read_data' module is in 'snRNAseq/code' relative to where the script is run
sys.path.append(os.path.join(os.path.dirname(__file__), "snRNAseq/code"))
try:
    from read_data import read_velmeshev, read_wang
except ImportError:
    print("Error: Could not import read_velmeshev or read_wang from snRNAseq/code/read_data.py")
    print("Please ensure the 'snRNAseq/code' directory exists and 'read_data.py' contains these functions.")
    sys.exit(1)


def integrate_snrnaseq_harmony(
    input_h5ad_paths: list[str],
    reading_functions: list, # Changed from dict to list
    output_dir: str = "snRNAseq/combined",
    output_filename: str = "integrated_harmony.h5ad",
    n_top_genes: int = 3000,
    n_comps: int = 20,
    max_iter_harmony: int = 30, # Maximum iterations for Harmony integration
    n_neighbors: int = 30,
    min_dist_umap: float = 0.3,
    random_state: int = 42,
    batch_key: str = 'chemistry',
    use_pearson_residuals: bool = True,
    plot_umaps: bool = True,
    plot_color_keys: list[str] = None,
    figure_size_inches: tuple[int, int] = (16, 10),
    age_log2_plot_settings: dict = None,
    perform_qc_plots: bool = True, # New parameter for QC plots
    mt_gene_prefix: str = 'MT-', # New parameter for mitochondrial gene prefix
) -> ad.AnnData:
    """
    Integrates single-nucleus RNA-seq (snRNA-seq) datasets using Harmony.

    This function reads multiple snRNA-seq datasets, performs initial QC and filtering,
    preprocesses them (either using Pearson residuals or a Seurat-like workflow),
    integrates them with Harmony, performs UMAP dimensionality reduction, and saves
    the integrated AnnData object. Optionally, it generates UMAP plots to visualize
    the integration and diagnostic QC plots.

    Args:
        input_h5ad_paths: A list of paths to the input .h5ad files.
        reading_functions: A list of functions, where each function corresponds to an
                           input file in `input_h5ad_paths` and is used to read and
                           preprocess that specific .h5ad file. The order must match.
                           Each function should take a file path as input and return an AnnData object.
        output_dir: Directory to save the integrated AnnData object and plots.
        output_filename: Filename for the integrated AnnData object.
        n_top_genes: Number of top variable genes to select during preprocessing.
                     Used with Pearson residuals.
        n_comps: Number of principal components to compute. Used for PCA before Harmony.
        n_neighbors: Number of neighbors for UMAP and PAGA.
        min_dist_umap: Minimum distance parameter for UMAP.
        random_state: Random seed for reproducibility.
        batch_key: The observation key in AnnData (.obs) that identifies the batch
                   (e.g., 'chemistry', 'dataset', 'origin') for Harmony integration.
        use_pearson_residuals: If True, uses `sc.experimental.pp.recipe_pearson_residuals`
                               for preprocessing. If False, uses a Seurat-like workflow
                               (`normalize_total`, `log1p`, `highly_variable_genes`, `scale`).
        plot_umaps: If True, generates and displays UMAP plots of the integrated data.
        plot_color_keys: A list of .obs keys to use for coloring the UMAP plots.
                         Defaults to ['origin', 'dataset', 'chemistry', 'age_log2', 'lineage', 'tissue'].
        figure_size_inches: Tuple specifying the width and height of the UMAP plot figure.
        age_log2_plot_settings: Dictionary containing settings for 'age_log2' colorbar.
                                Expected keys: 'age_ticks' (np.array). If None, default settings will be used.
        perform_qc_plots: If True, calculates and plots standard QC metrics (total counts, %MT).
        mt_gene_prefix: Prefix used to identify mitochondrial genes (e.g., 'MT-', 'mt-').

    Returns:
        An AnnData object containing the integrated and processed data.
    """

    if plot_color_keys is None:
        plot_color_keys = ['origin', 'dataset', 'chemistry', 'age_log2', 'lineage', 'tissue']
    if age_log2_plot_settings is None:
        age_log2_plot_settings = {'age_ticks': np.array([0, 1, 3, 5, 9, 25])}

    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists early

    # 1. Load and Concatenate Data
    print("Step 1: Loading and concatenating data...")
    adatas = [] # Changed to a list to store AnnData objects
    if len(input_h5ad_paths) != len(reading_functions):
        raise ValueError("The number of input_h5ad_paths must match the number of reading_functions.")

    for i, path in enumerate(input_h5ad_paths):
        read_func = reading_functions[i] # Get function by index
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        print(f"  Reading {path} using {read_func.__name__}...")
        adata = read_func(path)
        # Add a 'dataset' column to identify the origin for plotting/analysis later
        if 'dataset' not in adata.obs.columns:
            adata.obs['dataset'] = dataset_name
        adatas.append(adata)

    # Concatenate the list of AnnData objects
    adata_concat = ad.concat(adatas, merge='first', label='origin', index_unique="-")

    # The original code takes `data = data.raw.to_adata()`.
    # This implies that the raw data was stored and now we're converting it back to the main AnnData object.
    # We will ensure all relevant .obs and .var are copied.
    if adata_concat.raw is not None:
        data = adata_concat.raw.to_adata()
        data.obs = adata_concat.obs.copy()
        data.var = adata_concat.var.copy()
        data.uns = adata_concat.uns.copy()
        print("  Converted .raw to .X for primary operations.")
    else:
        warnings.warn("'.raw' attribute not found. Proceeding with '.X' as is. "
                      "Ensure '.X' contains raw counts if Pearson residuals are desired.")
        data = adata_concat

    print(f"Combined data dimensions: {data.shape}")

    # Check for requested color keys and print all .obs columns
    print("\n--- .obs columns in concatenated data ---")
    all_obs_columns = data.obs.columns.tolist()
    print(all_obs_columns)
    print("------------------------------------------")

    # Filter plot_color_keys to only include existing columns
    valid_plot_color_keys = []
    for key in plot_color_keys:
        if key in all_obs_columns:
            valid_plot_color_keys.append(key)
        else:
            warnings.warn(f"Plot color key '{key}' not found in data.obs. "
                          f"Skipping this key for plotting. You may need to update your reading functions "
                          f"to include this column in .obs.")
    plot_color_keys = valid_plot_color_keys


    # 2. Basic QC and Diagnostic Plots (if requested)
    if perform_qc_plots:
        print("\nStep 2: Performing basic QC and generating diagnostic plots...")
        try:
            # Calculate QC metrics
            # Total counts per cell
            data.obs['total_counts'] = data.X.sum(axis=1).A1 if hasattr(data.X, 'A1') else data.X.sum(axis=1)

            # Number of genes expressed per cell
            data.obs['n_genes_by_counts'] = (data.X > 0).sum(axis=1).A1 if hasattr(data.X, 'A1') else (data.X > 0).sum(axis=1)

            # Mitochondrial gene percentage
            if any(gene.startswith(mt_gene_prefix) for gene in data.var_names):
                data.var['mt'] = data.var_names.str.startswith(mt_gene_prefix)
                sc.pp.calculate_qc_metrics(data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
                print("  Calculated % mitochondrial counts.")
            else:
                warnings.warn(f"No mitochondrial genes found with prefix '{mt_gene_prefix}'. Skipping %MT calculation.")
                data.obs['pct_counts_mt'] = 0 # Add a placeholder column

            # Plotting QC metrics
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle("Pre-integration QC Metrics", fontsize=16)

            # Violin plot for n_genes_by_counts
            sc.pl.violin(data, 'n_genes_by_counts', jitter=0.4, ax=axes[0], show=False)
            axes[0].set_title('Genes by Counts')

            # Violin plot for total_counts
            sc.pl.violin(data, 'total_counts', jitter=0.4, ax=axes[1], show=False)
            axes[1].set_title('Total Counts')

            # Violin plot for pct_counts_mt
            if 'pct_counts_mt' in data.obs.columns:
                sc.pl.violin(data, 'pct_counts_mt', jitter=0.4, ax=axes[2], show=False)
                axes[2].set_title('Mitochondrial %')
            else:
                axes[2].set_visible(False) # Hide if no MT data

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
            qc_plot_path = os.path.join(output_dir, "pre_integration_qc_plots.png")
            plt.savefig(qc_plot_path, dpi=300)
            plt.close(fig)
            print(f"  QC plots saved to {qc_plot_path}")

        except Exception as e:
            warnings.warn(f"Failed to generate QC plots: {e}. Continuing without QC plots.")

    # 3. Preprocessing for Harmony
    print(f"\nStep 3: Preprocessing data using {'Pearson residuals' if use_pearson_residuals else 'Seurat-like method'}...")
    if use_pearson_residuals:
        try:
            sc.experimental.pp.recipe_pearson_residuals(
                data,
                n_top_genes=n_top_genes,
                batch_key=batch_key,
                random_state=random_state,
                n_comps=n_comps,
                inplace=True
            )
            print(f"  Data now contains `X_pca` based on Pearson residuals with {data.obsm['X_pca'].shape[1]} components.")
        except Exception as e:
            raise RuntimeError(f"Error during Pearson residuals preprocessing: {e}. "
                               "Ensure data.X is suitable for this method and batch_key is valid.")
    else:
        print("  Using traditional Seurat-like preprocessing (normalize, log1p, HVG, scale, PCA)...")
        try:
            sc.pp.normalize_total(data, target_sum=1e4)
            sc.pp.log1p(data)
            sc.pp.highly_variable_genes(data, flavor='seurat', min_mean=0.1, max_mean=8, min_disp=0.5)
            hvg_genes = data.var['highly_variable']
            data = data[:, hvg_genes].copy()
            sc.pp.scale(data)
            sc.pp.pca(data, n_comps=n_comps, random_state=random_state)
            print(f"  Data now contains `X_pca` from Seurat-like method with {data.obsm['X_pca'].shape[1]} components.")
        except Exception as e:
            raise RuntimeError(f"Error during Seurat-like preprocessing: {e}.")

    # 4. Harmony Integration
    print(f"\nStep 4: Integrating data with Harmony using '{batch_key}' as batch key...")
    if batch_key not in data.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in data.obs. Cannot perform Harmony integration.")
    if 'X_pca' not in data.obsm:
        raise ValueError(f"'X_pca' not found in data.obsm. PCA must be run before Harmony integration.")

    try:
        sce.pp.harmony_integrate(data, key=batch_key, max_iter_harmony=max_iter_harmony, basis='X_pca', adjusted_basis='X_harmony')
        print(f"  Harmony integration complete. Integrated embedding in `data.obsm['X_harmony']`.")
    except Exception as e:
        raise RuntimeError(f"Error during Harmony integration: {e}. "
                           "Ensure 'X_pca' exists in data.obsm and the batch_key column is valid.")

    # 5. Neighborhood Graph and UMAP
    print("\nStep 5: Computing neighbors and UMAP...")
    if 'X_harmony' not in data.obsm:
        raise ValueError(f"'X_harmony' not found in data.obsm. Harmony integration must complete successfully.")
    try:
        sc.pp.neighbors(data, n_neighbors=n_neighbors, use_rep="X_harmony", random_state=random_state)
        sc.tl.umap(data, min_dist=min_dist_umap, n_components=2, random_state=random_state)
        print("  UMAP embedding computed and stored in `data.obsm['X_umap']`.")
    except Exception as e:
        raise RuntimeError(f"Error during neighbor graph or UMAP computation: {e}.")


    # 6. Optional: Further Analysis (e.g., Clustering)
    print("\nStep 6: Performing Leiden clustering...")
    try:
        sc.tl.leiden(data, random_state=random_state, key_added='leiden_clusters')
        print(f"  Leiden clustering complete. Clusters in `data.obs['leiden_clusters']`.")
        # Add leiden_clusters to plot_color_keys if not already there and valid
        # if 'leiden_clusters' not in plot_color_keys:
            # plot_color_keys.append('leiden_clusters')
    except Exception as e:
        warnings.warn(f"Failed to perform Leiden clustering: {e}. Continuing without clustering results.")

    # 7. Save Processed Data
    print(f"\nStep 7: Saving integrated AnnData object to {output_dir}/{output_filename}...")
    output_path = os.path.join(output_dir, output_filename)
    try:
        data.write_h5ad(output_path)
        print("  AnnData object saved successfully.")
    except Exception as e:
        warnings.warn(f"Failed to save AnnData object: {e}. This might indicate a permissions or disk space issue.")

    # 8. Plotting UMAPs
    if plot_umaps and plot_color_keys:
        print("\nStep 8: Generating UMAP plots...")
        n_plots = len(plot_color_keys)
        if n_plots == 0:
            print("  No valid plot_color_keys to plot. Skipping UMAP plotting.")
            return data

        ncols = 3
        nrows = int(np.ceil(n_plots / ncols))
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figure_size_inches)

        # Ensure axes is always a flat iterable
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, key in enumerate(plot_color_keys):
            ax = axes[i]
            # Use 'viridis' for continuous keys like 'age_log2', 'Spectral' for discrete
            cmap = 'Spectral_r' if data.obs[key].dtype == 'float64' or key == 'age_log2' else 'Spectral'
            try:
                sc.pl.umap(data, color=key, show=False, ax=ax, color_map=cmap)
                ax.set_title(key)
            except Exception as e:
                warnings.warn(f"Failed to plot UMAP for '{key}': {e}. Skipping this plot.")
                ax.set_visible(False) # Hide the subplot if plotting fails

        # Hide any remaining empty subplots
        for i in range(len(plot_color_keys), len(axes)):
            axes[i].set_visible(False)

        # Custom ticks for 'age_log2' if present
        if 'age_log2' in plot_color_keys and 'age_ticks' in age_log2_plot_settings:
            try:
                # Find the axis that plotted 'age_log2'
                age_ax_index = plot_color_keys.index('age_log2')
                if age_ax_index < len(axes) and axes[age_ax_index].get_title() == 'age_log2':
                    # Find the colorbar associated with this specific axis
                    cbar_axes = [a for a in fig.axes if hasattr(a, '_colorbar') and a._colorbar.ax.get_figure() == fig and a._colorbar.ax.get_ylabel() != '']
                    # This is still heuristic, but tries to get the colorbar of the correct plot.
                    # A more robust solution for specific colorbar control involves plotting 'age_log2' in a dedicated call if necessary.
                    if cbar_axes:
                         # Attempt to find the colorbar most likely associated with the 'age_log2' plot
                         # This might be the last colorbar added or the one closest to the 'age_log2' subplot.
                         # As a fallback, we'll try the first one that appears to be a colorbar.
                        cbar_ax_for_age = cbar_axes[0]
                        for cbar_candidate in cbar_axes:
                            # Heuristic: check if the colorbar's parent axis is the age_log2 plot's axis
                            if cbar_candidate._colorbar.ax.get_ylabel() == 'age_log2' or 'age_log2' in cbar_candidate._colorbar.ax.get_title():
                                cbar_ax_for_age = cbar_candidate._colorbar.ax
                                break

                        log_ticks = np.log2(age_log2_plot_settings['age_ticks'] + 1)
                        cbar_ax_for_age.set_yticks(log_ticks)
                        cbar_ax_for_age.set_yticklabels([str(x) for x in age_log2_plot_settings['age_ticks']])
                        cbar_ax_for_age.set_ylabel("Age (Years)")
                        print("  Set custom ticks for 'age_log2' colorbar.")
                else:
                    warnings.warn("Could not reliably identify 'age_log2' plot axis for custom colorbar ticks.")
            except Exception as e:
                warnings.warn(f"Failed to set custom 'age_log2' ticks: {e}. Continuing without custom ticks.")

        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout for suptitle
        plt.suptitle("Harmony Integrated UMAPs", y=1.0, fontsize=16)
        plot_output_path = os.path.join(output_dir, "integrated_harmony_umaps.png")
        plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        print(f"  UMAP plots saved to {plot_output_path}.")
    elif plot_umaps:
        print("  UMAP plotting requested, but no valid plot_color_keys found. Skipping UMAP plots.")
    else:
        print("  Skipping UMAP plotting as requested.")

    print("\nPipeline complete.")
    return data


if __name__ == "__main__":
    # --- Configuration for your run ---
    # Define paths to your input files
    # IMPORTANT: Adjust these paths to your actual data on the HPC.
    # Relative paths might work if your SLURM script is run from the correct directory.
    # Absolute paths are generally safer on HPC.
    velmeshev_path = "snRNAseq/data/velmeshev.h5ad"
    wang_path = "snRNAseq/data/wang.h5ad"
    # Add more paths as needed
    # dataset3_path = "path/to/your/dataset3.h5ad"

    # Define your reading functions.
    # These functions must be defined in 'snRNAseq/code/read_data.py'
    # and should take a file path as input and return an AnnData object.
    h5ad_reading_functions = [
        read_velmeshev,
        read_wang,
        # read_dataset3, # Add if you have dataset3 and its reading function
    ]

    # List of all input .h5ad file paths to process
    input_files_to_process = [velmeshev_path, wang_path]
    # input_files_to_process = [velmeshev_path, wang_path, dataset3_path] # Example for 3 files

    # Output settings
    output_base_dir = "snRNAseq/outputs" # A top-level directory for your results
    run_name = "velmeshev-wang_pcs15" # A unique name for this specific run
    output_specific_dir = os.path.join(output_base_dir, run_name)
    output_h5ad_filename = f"{run_name}.h5ad"

    # Plotting preferences
    plot_keys_for_umap = ['origin', 'dataset', 'chemistry', 'age_log2', 'lineage', 'tissue']
    age_plot_settings = {'colorbar_index': 0, 'age_ticks': np.array([0, 1, 3, 5, 9, 25])}

    print(f"Starting integration pipeline for run: {run_name}")
    try:
        integrated_adata = integrate_snrnaseq_harmony(
            input_h5ad_paths=input_files_to_process,
            reading_functions=h5ad_reading_functions, # Now a list
            output_dir=output_specific_dir,
            output_filename=output_h5ad_filename,
            n_top_genes=3000,
            n_comps=15,
            max_iter_harmony=50, # Maximum iterations for Harmony integration
            n_neighbors=30,
            min_dist_umap=0.3,
            random_state=42,
            batch_key='chemistry',
            use_pearson_residuals=True,
            plot_umaps=True,
            plot_color_keys=plot_keys_for_umap,
            figure_size_inches=(20, 15),
            age_log2_plot_settings=age_plot_settings,
            perform_qc_plots=True, # Enable QC plots
            mt_gene_prefix='MT-' # Adjust if your mitochondrial genes start with something else
        )
        print(f"\nIntegration successfully completed for {run_name}.")
        print(f"Integrated AnnData object saved to: {os.path.join(output_specific_dir, output_h5ad_filename)}")
        print(f"UMAP plots saved to: {os.path.join(output_specific_dir, 'integrated_harmony_umaps.png')}")
        print("Final AnnData object structure:")
        print(integrated_adata)
        print(integrated_adata.obs.head())

    except FileNotFoundError as e:
        print(f"ERROR: A required input file was not found. Please check paths. Details: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: Configuration or data value issue. Details: {e}")
        sys.exit(1)
    except RuntimeError as e: # Catch specific runtime errors now
        print(f"ERROR: A critical processing step failed. Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected ERROR occurred during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)