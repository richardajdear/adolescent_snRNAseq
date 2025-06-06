import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import anndata as ad

def plot_integration_umaps(
    adata: ad.AnnData,
    umap_key_to_plot: str,
    plot_color_keys: list[str],
    output_dir: str,
    output_filename_prefix: str,
    figure_size_inches: tuple[int, int],
    age_log2_plot_settings: dict,
    plot_title_suffix: str = "",
    add_suptitle: bool = False, # New parameter: default to False (no suptitle)
):
    """
    Generates and saves UMAP plots for a specified UMAP embedding (e.g., 'X_umap_scvi').

    This function temporarily sets the specified UMAP embedding to adata.obsm['X_umap']
    for plotting, then restores the original 'X_umap_harmony' (if it exists) or
    cleans up. This is a workaround for sc.pl.umap issues with non-default bases.

    Args:
        adata: The AnnData object.
        umap_key_to_plot: The exact key in adata.obsm to use for plotting UMAP
                          (e.g., 'X_umap_scvi', 'X_umap_scanvi', 'X_umap_harmony').
        plot_color_keys: A list of .obs keys to use for coloring the UMAP plots.
        output_dir: Directory to save the plots.
        output_filename_prefix: Prefix for the output image file (e.g., 'integrated_scvi_umaps').
        figure_size_inches: Tuple specifying the width and height of the UMAP plot figure.
        age_log2_plot_settings: Dictionary containing settings for 'age_log2' colorbar.
        plot_title_suffix: Additional text to append to the plot's main title (e.g., "scVI").
                           This is used for both subplot titles and the main suptitle.
        add_suptitle: If True, a main suptitle will be added to the figure.
                      Defaults to False.
    """
    if not plot_color_keys:
        print(f"  No valid plot_color_keys to plot for {umap_key_to_plot}. Skipping plotting.")
        return

    # Store original X_umap if it exists, to restore later
    original_x_umap = None
    if 'X_umap' in adata.obsm:
        original_x_umap = adata.obsm['X_umap'].copy()

    # Temporarily set adata.obsm['X_umap'] to the desired embedding for plotting
    if umap_key_to_plot not in adata.obsm:
        warnings.warn(f"Embedding key '{umap_key_to_plot}' not found in adata.obsm. Cannot plot.")
        return

    adata.obsm['X_umap'] = adata.obsm[umap_key_to_plot].copy()
    print(f"  Temporarily set adata.obsm['X_umap'] to '{umap_key_to_plot}' for plotting.")

    n_plots = len(plot_color_keys)
    ncols = 3
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figure_size_inches)

    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Iterate through plot_color_keys and plot
    for i, key in enumerate(plot_color_keys):
        if i >= len(axes): # Safety check if more keys than subplots unexpectedly
            break
        ax = axes[i]
        cmap = 'Spectral_r' if adata.obs[key].dtype == 'float64' or key == 'age_log2' else 'Spectral'
        try:
            # Plot using the temporary X_umap
            sc.pl.umap(adata, color=key, show=False, ax=ax, color_map=cmap)
            ax.set_title(f"{plot_title_suffix} UMAP - {key}".strip()) # .strip() to clean up leading space if suffix is empty
        except Exception as e:
            warnings.warn(f"Failed to plot UMAP for '{key}' on '{umap_key_to_plot}': {e}. Skipping this subplot.")
            ax.set_visible(False) # Hide the subplot if plotting fails

    # Hide any remaining empty subplots
    for i in range(len(plot_color_keys), len(axes)):
        if i < len(axes): # Ensure index is valid
            axes[i].set_visible(False)

    # Custom ticks for 'age_log2' if present
    if 'age_log2' in plot_color_keys and 'age_ticks' in age_log2_plot_settings:
        try:
            # Find the axis that plotted 'age_log2'
            age_ax_index = -1
            for idx, plot_key in enumerate(plot_color_keys):
                if plot_key == 'age_log2':
                    age_ax_index = idx
                    break

            if age_ax_index != -1 and age_ax_index < len(axes) and axes[age_ax_index].get_title().endswith('age_log2'):
                # Heuristically find the colorbar for the specific subplot
                cbar_ax = None
                for a_fig in fig.axes: # Iterate through all axes in the figure
                    if hasattr(a_fig, 'get_ylabel') and 'age_log2' in a_fig.get_ylabel():
                         cbar_ax = a_fig
                         break
                # Fallback: find any colorbar if the above fails
                if cbar_ax is None:
                    for a_fig in fig.axes:
                        if hasattr(a_fig, 'get_label') and 'colorbar' in a_fig.get_label():
                            cbar_ax = a_fig
                            break

                if cbar_ax:
                    log_ticks = np.log2(age_log2_plot_settings['age_ticks'] + 1)
                    cbar_ax.set_yticks(log_ticks)
                    cbar_ax.set_yticklabels([str(x) for x in age_log2_plot_settings['age_ticks']])
                    cbar_ax.set_ylabel("Age (Years)")
                    print("  Set custom ticks for 'age_log2' colorbar.")
                else:
                    warnings.warn("Could not reliably identify 'age_log2' colorbar axis for custom ticks.")
            else:
                warnings.warn("Could not reliably identify 'age_log2' plot axis for custom colorbar ticks.")
        except Exception as e:
            warnings.warn(f"Failed to set custom 'age_log2' ticks: {e}. Continuing without custom ticks.")

    # Only add suptitle if add_suptitle is True
    if add_suptitle:
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout for suptitle
        plt.suptitle(f"{plot_title_suffix} Integrated UMAPs".strip(), y=1.0, fontsize=16)
    else:
        plt.tight_layout() # Use standard tight_layout if no suptitle

    plot_output_path = os.path.join(output_dir, f"{output_filename_prefix}.png")
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    print(f"  UMAP plots for {umap_key_to_plot} saved to {plot_output_path}.")

    # Restore original X_umap
    if original_x_umap is not None:
        adata.obsm['X_umap'] = original_x_umap
        print(f"  Restored original adata.obsm['X_umap'].")
    elif 'X_umap' in adata.obsm: # If it was created by us but no original existed
        del adata.obsm['X_umap'] # Clean up the temporary X_umap if no original existed
        print(f"  Cleaned up temporary adata.obsm['X_umap'].")