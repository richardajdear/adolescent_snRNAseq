import scanpy as sc
import pandas as pd
import numpy as np
import os
import warnings
from scipy.sparse import issparse
import re
import json

# --- Define utility functions at the top level ---
def sanitize_col_name(name: str) -> str:
    """Replaces characters that are problematic for HDF5 paths (like / or \)."""
    return re.sub(r'[^\w\-]', '_', str(name))

def contains_problematic_chars(name: str) -> bool:
    """Checks if a string contains characters problematic for HDF5 paths (like / or \)."""
    return bool(re.search(r'[/\\]', str(name)))

def aggregate_to_metacells(
    original_adata_path: str,
    metacell_assignments_path: str,
    raw_counts_layer_key: str = 'counts',
    quantitative_obs_cols: list = None,
    categorical_obs_cols: list = None
):
    """
    Aggregates raw counts and specified .obs fields from single cells into metacells.

    Args:
        original_adata_path: Path to the original AnnData file.
        metacell_assignments_path: Path to the .h5ad file containing metacell assignments.
        raw_counts_layer_key: The key in adata.layers where raw (integer) counts are stored.
                              Set to None if adata.X is raw counts.
        quantitative_obs_cols: List of column names in original_adata.obs to aggregate
                               quantitatively (mean, std, min, max, median, IQR).
        categorical_obs_cols: List of column names in original_adata.obs to aggregate
                              categorically (majority vote, proportions, entropy, unique categories).
                              Columns containing '/' or '\' in their original *name* will be skipped.
    
    Returns:
        AnnData: The newly created AnnData object containing aggregated metacell data.
    """

    print(f"Loading original AnnData from: {original_adata_path}")
    original_adata = sc.read_h5ad(original_adata_path)

    print(f"Loading metacell assignments from: {metacell_assignments_path}")
    metacell_assignments_adata = sc.read_h5ad(metacell_assignments_path)

    # --- 1. Prepare raw counts source ---
    if raw_counts_layer_key and raw_counts_layer_key in original_adata.layers:
        print(f"Using raw counts from adata.layers['{raw_counts_layer_key}'] for aggregation.")
        raw_counts_matrix = original_adata.layers[raw_counts_layer_key]
    elif not raw_counts_layer_key:
        print("Using adata.X for raw counts aggregation (raw_counts_layer_key was None). "
              "Ensure adata.X contains raw counts if desired.")
        raw_counts_matrix = original_adata.X
    else:
        raise ValueError(f"Raw counts layer '{raw_counts_layer_key}' not found in original_adata.layers "
                         "and raw_counts_layer_key was not None. Please specify correct layer or set to None.")

    if not (issparse(raw_counts_matrix) or raw_counts_matrix.dtype in [np.int32, np.int64, np.float32, np.float64]):
        warnings.warn("Raw counts source is not a sparse matrix or a recognized numeric type. "
                      "Ensure it's appropriate for summation (e.g., integer counts or floats from normalized data).")
    
    # --- 2. Robust Alignment of Metacell Assignments ---
    if 'metacell' not in metacell_assignments_adata.obs.columns:
        if metacell_assignments_adata.obs.shape[1] == 1:
            metacell_assignments_adata.obs.columns = ['metacell']
            print("Renamed the single column in assignments_adata.obs to 'metacell'.")
        else:
            raise ValueError("Metacell assignments AnnData must have a column named 'metacell' in .obs, "
                             "or be a 1-column DataFrame for auto-naming.")

    print("Attempting to align cell IDs for aggregation...")

    if original_adata.obs_names.equals(metacell_assignments_adata.obs_names):
        print("Cell IDs (obs_names) in original_adata and assignments_adata perfectly match. Proceeding directly.")
        metacell_assignments_aligned_df = metacell_assignments_adata.obs[['metacell']].copy()
    
    elif original_adata.n_obs == metacell_assignments_adata.n_obs:
        print(f"Number of cells match ({original_adata.n_obs}), but obs_names differ. Assuming positional alignment.")
        metacell_assignments_aligned_df = pd.DataFrame(
            metacell_assignments_adata.obs['metacell'].values,
            index=original_adata.obs_names,
            columns=['metacell']
        )
    
    else:
        print(f"Warning: Cell counts differ ({original_adata.n_obs} vs {metacell_assignments_adata.n_obs}) "
              "and obs_names do not match. Attempting reindex, which may filter cells.")
        metacell_assignments_aligned_df = metacell_assignments_adata.obs.reindex(original_adata.obs_names)[['metacell']].copy()

        cells_with_assignments_mask = metacell_assignments_aligned_df['metacell'].notna()
        if cells_with_assignments_mask.sum() == 0:
             raise ValueError("After reindexing/alignment, 0 cells have valid metacell assignments. "
                              "Please check cell IDs and ensure overlap between original_adata and assignments_adata.")
        original_adata = original_adata[cells_with_assignments_mask, :].copy()
        metacell_assignments_aligned_df = metacell_assignments_aligned_df[cells_with_assignments_mask]
        raw_counts_matrix = raw_counts_matrix[cells_with_assignments_mask, :]
        
        print(f"Filtered to {original_adata.n_obs} cells with valid metacell assignments after reindex.")

    original_adata.obs['metacell_id'] = metacell_assignments_aligned_df['metacell'].astype('category')
    
    unique_metacells = original_adata.obs['metacell_id'].cat.categories
    num_metacells = len(unique_metacells)
    
    print(f"Identified {num_metacells} unique metacells for aggregation.")

    # --- 3. Prepare for Aggregation ---
    aggregated_expression_rows = []
    aggregated_obs_data = []

    # --- 4. Perform Aggregation ---
    print("Starting aggregation for gene counts and metadata...")
    
    # Filter categorical_obs_cols for problematic names before processing
    process_categorical_cols = []
    if categorical_obs_cols:
        for col in categorical_obs_cols:
            if contains_problematic_chars(col):
                warnings.warn(f"Categorical column name '{col}' contains problematic characters (like / or \). "
                              "This column will be skipped for aggregation to prevent H5AD saving errors.")
            else:
                process_categorical_cols.append(col)
    
    for i, mc_id in enumerate(unique_metacells):
        cells_in_mc_mask = original_adata.obs['metacell_id'] == mc_id
        
        if cells_in_mc_mask.sum() == 0:
            print(f"Warning: Metacell {mc_id} has no assigned cells after filtering. Skipping.")
            continue

        current_mc_obs = original_adata.obs[cells_in_mc_mask]
        current_mc_raw_counts = raw_counts_matrix[cells_in_mc_mask, :]

        # a. Aggregate Raw Counts (Sum)
        if issparse(current_mc_raw_counts):
            mc_sum_counts = current_mc_raw_counts.sum(axis=0).A1
        else:
            mc_sum_counts = current_mc_raw_counts.sum(axis=0)
        aggregated_expression_rows.append(mc_sum_counts)

        # b. Aggregate Metadata
        metacell_obs_entry = {'metacell_id': mc_id, 'metacell_size': current_mc_obs.shape[0]}

        # Quantitative columns
        for col in quantitative_obs_cols or []:
            if col in original_adata.obs.columns:
                values = current_mc_obs[col].dropna()
                if not values.empty:
                    metacell_obs_entry[f'mean_{col}'] = values.mean()
                    metacell_obs_entry[f'std_{col}'] = values.std() if len(values) > 1 else 0.0
                    metacell_obs_entry[f'min_{col}'] = values.min()
                    metacell_obs_entry[f'max_{col}'] = values.max()
                    metacell_obs_entry[f'median_{col}'] = values.median()
                    if len(values) > 1:
                        q1 = values.quantile(0.25)
                        q3 = values.quantile(0.75)
                        metacell_obs_entry[f'iqr_{col}'] = q3 - q1
                    else:
                        metacell_obs_entry[f'iqr_{col}'] = np.nan
                else:
                    metacell_obs_entry[f'mean_{col}'] = np.nan
                    metacell_obs_entry[f'std_{col}'] = np.nan
                    metacell_obs_entry[f'min_{col}'] = np.nan
                    metacell_obs_entry[f'max_{col}'] = np.nan
                    metacell_obs_entry[f'median_{col}'] = np.nan
                    metacell_obs_entry[f'iqr_{col}'] = np.nan
            else:
                warnings.warn(f"Quantitative column '{col}' not found in original_adata.obs. Skipping.")

        # Categorical columns (using the filtered list)
        for col in process_categorical_cols:
            if col in original_adata.obs.columns:
                value_counts = current_mc_obs[col].value_counts(normalize=False, dropna=True)
                total_cells_for_category = value_counts.sum()

                if total_cells_for_category > 0:
                    metacell_obs_entry[f'majority_{col}'] = value_counts.index[0]
                    metacell_obs_entry[f'purity_{col}'] = value_counts.max() / total_cells_for_category
                    metacell_obs_entry[f'unique_categories_{col}'] = current_mc_obs[col].nunique()

                    proportions_series = (value_counts / total_cells_for_category)
                    sanitized_proportions_dict = {
                        sanitize_col_name(k): round(v, 3) for k, v in proportions_series.items()
                    }
                    metacell_obs_entry[f'proportions_{col}'] = json.dumps(sanitized_proportions_dict)

                    p_i = proportions_series.values
                    metacell_obs_entry[f'entropy_{col}'] = -np.sum(p_i * np.log(p_i + 1e-10))
                else:
                    metacell_obs_entry[f'majority_{col}'] = np.nan
                    metacell_obs_entry[f'purity_{col}'] = 0.0
                    metacell_obs_entry[f'unique_categories_{col}'] = 0
                    metacell_obs_entry[f'proportions_{col}'] = "{}"
                    metacell_obs_entry[f'entropy_{col}'] = 0.0

            else:
                warnings.warn(f"Categorical column '{col}' not found in original_adata.obs. Skipping.")
        
        aggregated_obs_data.append(metacell_obs_entry)

    # --- 5. Construct the final metacell AnnData object ---
    aggregated_X_matrix = np.array(aggregated_expression_rows)
    
    aggregated_obs_df = pd.DataFrame(aggregated_obs_data)
    
    if 'metacell_id' not in aggregated_obs_df.columns:
        raise KeyError("Internal Error: 'metacell_id' column was not created in aggregated_obs_data.")
    aggregated_obs_df = aggregated_obs_df.set_index('metacell_id')
    
    metacell_adata = sc.AnnData(
        X=aggregated_X_matrix,
        obs=aggregated_obs_df,
        var=original_adata.var.copy()
    )

    print("\nAggregation complete.")
    print(f"Final aggregated metacell AnnData shape: {metacell_adata.shape}")
    
    # --- Print Summary Table for General Metrics ---
    print("\n--- Summary Statistics for Metacell Metrics (Metacell-level) ---")
    general_summary_data = []

    # Calculate total gene counts per metacell
    total_gene_counts_per_metacell = metacell_adata.X.sum(axis=1)
    
    # Calculate total non-zero genes per metacell
    if issparse(metacell_adata.X):
        total_nonzero_genes_per_metacell = metacell_adata.X.getnnz(axis=1)
    else:
        total_nonzero_genes_per_metacell = (metacell_adata.X > 0).sum(axis=1)

    summary_metrics_data = {
        'Number of cells per metacell': metacell_adata.obs['metacell_size'],
        'Total gene counts per metacell': pd.Series(total_gene_counts_per_metacell, index=metacell_adata.obs_names),
        'Total non-zero genes per metacell': pd.Series(total_nonzero_genes_per_metacell, index=metacell_adata.obs_names)
    }

    for metric_name, series in summary_metrics_data.items():
        if not series.empty:
            general_summary_data.append({
                'Metric': metric_name,
                'Min': series.min(),
                'Mean': series.mean(),
                'Max': series.max(),
                'Q25': series.quantile(0.25),
                'Median': series.median(),
                'Q75': series.quantile(0.75),
            })
        else:
            general_summary_data.append({
                'Metric': metric_name,
                'Min': np.nan, 'Mean': np.nan, 'Max': np.nan,
                'Q25': np.nan, 'Median': np.nan, 'Q75': np.nan,
            })
    
    if general_summary_data:
        general_summary_df = pd.DataFrame(general_summary_data).round(3)
        print(general_summary_df.to_string(index=False))
    else:
        print("No general metacell metrics to summarize.")
    print("------------------------------------------------------------------")


    # --- Print Summary Table for Categorical Variables ---
    print("\n--- Summary Statistics for Categorical Variables (Metacell-level) ---")
    summary_data = []
    for col in process_categorical_cols:
        if f'entropy_{col}' in metacell_adata.obs.columns:
            entropy_series = metacell_adata.obs[f'entropy_{col}'].dropna()
            purity_series = metacell_adata.obs[f'purity_{col}'].dropna()
            unique_cats_series = metacell_adata.obs[f'unique_categories_{col}'].dropna()

            if not entropy_series.empty:
                summary_data.append({
                    'Variable': col,
                    'Entropy_Min': entropy_series.min(),
                    'Entropy_Mean': entropy_series.mean(),
                    'Entropy_Max': entropy_series.max(),
                    'Purity_Min': purity_series.min(),
                    'Purity_Mean': purity_series.mean(),
                    'Purity_Max': purity_series.max(),
                    'UniqueCategories_Min': unique_cats_series.min(),
                    'UniqueCategories_Mean': unique_cats_series.mean(),
                    'UniqueCategories_Max': unique_cats_series.max(),
                })
            else:
                 summary_data.append({
                    'Variable': col,
                    'Entropy_Min': np.nan, 'Entropy_Mean': np.nan, 'Entropy_Max': np.nan,
                    'Purity_Min': np.nan, 'Purity_Mean': np.nan, 'Purity_Max': np.nan,
                    'UniqueCategories_Min': np.nan, 'UniqueCategories_Mean': np.nan, 'UniqueCategories_Max': np.nan,
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data).round(3)
        print(summary_df.to_string(index=False))
    else:
        print("No categorical columns were processed or found with valid data to summarize.")

    print("------------------------------------------------------------------")


    # --- 7. Construct and Save the aggregated metacell AnnData object ---
    # Derive output_aggregated_adata_path automatically
    # Example: /home/rajd2/rds/hpc-work/snRNAseq/outputs/velmeshev-wang_pcs20/velmeshev-wang_pcs20.h5ad
    # Target:  /home/rajd2/rds/hpc-work/snRNAseq/outputs/velmeshev-wang_pcs20/velmeshev-wang_pcs20_10000metacell_aggregated.h5ad

    original_base_dir = os.path.dirname(original_adata_path)
    original_filename_base = os.path.splitext(os.path.basename(original_adata_path))[0] # velmeshev-wang_pcs20

    # Extract metacell number from assignments_path
    # E.g., .../velmeshev-wang_pcs20_Xharmony_10000metacell_ids.h5ad
    # Regex to find Nmetacell before _ids.h5ad
    match = re.search(r'(\d+)metacell_ids\.h5ad$', os.path.basename(metacell_assignments_path))
    if match:
        metacell_num_str = match.group(1) # '10000'
    else:
        warnings.warn("Could not extract metacell number from assignment path. Using 'unknown_metacell_num'.")
        metacell_num_str = "unknown_metacell_num"
    
    # Construct the desired output filename
    output_filename = f"{original_filename_base}_{metacell_num_str}metacell_aggregated.h5ad"
    output_aggregated_adata_path = os.path.join(original_base_dir, output_filename)

    output_dir = os.path.dirname(output_aggregated_adata_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    metacell_adata.write_h5ad(output_aggregated_adata_path)
    print(f"Aggregated metacell data saved to: {output_aggregated_adata_path}")

    return metacell_adata

# --- Example Usage (How you might call this in your run_metaq.py or a separate script) ---
if __name__ == '__main__':
    # Define your actual paths for testing
    example_original_adata_path = "/home/rajd2/rds/hpc-work/snRNAseq/outputs/velmeshev-wang_pcs20/velmeshev-wang_pcs20.h5ad"
    example_metacell_assignment_path = "/home/rajd2/rds/hpc-work/snRNAseq/outputs/velmeshev-wang_pcs20/metaq/save/velmeshev-wang_pcs20_Xharmony_10000metacell_ids.h5ad"
    # output_aggregated_adata_path is now constructed automatically inside the function!

    example_quantitative_cols = ["age_years", "age_log2", "total_counts", "n_genes_by_counts"]
    example_categorical_cols = ["origin", "dataset", "chemistry", "lineage", "region", "tissue", "donor_id", "sex", "development_stage"]
    
    example_raw_counts_layer_key = None # Adjust based on your data


    print("--- Starting Metacell Aggregation Example ---")
    try:
        aggregated_metacells_result = aggregate_to_metacells(
            original_adata_path=example_original_adata_path,
            metacell_assignments_path=example_metacell_assignment_path,
            raw_counts_layer_key=example_raw_counts_layer_key,
            quantitative_obs_cols=example_quantitative_cols,
            categorical_obs_cols=example_categorical_cols
        )
        print("\n--- Aggregation Example Finished Successfully ---")

    except Exception as e:
        print(f"\n--- Aggregation Example Failed: {e} ---")
        import traceback
        traceback.print_exc()