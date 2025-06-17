# my_metaq_script.py
import torch
import numpy
import argparse
import os
import time
import scanpy as sc

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
print(f"NumPy version: {numpy.__version__}")
print(f"MetaQ-sc installed: {'MetaQ_sc' in globals()}")


from MetaQ_sc.main import run_metaq # Corrected import from previous steps

# Initialize variable to hold the path to the intermediate Harmony file if created
harmony_temp_file_path = None

try:
    start_time = time.time()

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run MetaQ-sc on single-cell RNA-seq data.")
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Full path to the input H5AD data file (e.g., /path/to/data/filename.h5ad)."
    )
    parser.add_argument(
        "--metacell_num",
        type=int,
        default=1000,
        help="Number of metacells to generate."
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="embedding",
        choices=["RNA", "ADT", "ATAC", "harmony", "embedding"],
        help="Type of data being processed (e.g., RNA, ADT, ATAC, harmony, embedding)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run MetaQ-sc on (cuda or cpu)."
    )
    parser.add_argument(
        "--prepare_harmony",
        action="store_true",
        help="If set, load the input .h5ad, extract X_harmony, save it as a new .h5ad, and use that for MetaQ."
    )


    args = parser.parse_args()

    # Determine the actual data file to pass to run_metaq
    data_file_for_metaq = args.data_file

    # --- Conditional Harmony Preparation ---
    if args.prepare_harmony:
        print("\n--- Preparing Harmony Embedding for MetaQ ---")
        if args.data_type not in ["harmony", "embedding"]:
            print(f"Warning: --prepare_harmony is set, but --data_type is '{args.data_type}'. "
                  "It is highly recommended to use 'harmony' or 'embedding' data_type.")

        # Load the original AnnData object
        print(f"Loading original data from: {args.data_file}")
        original_adata = sc.read_h5ad(args.data_file)

        if 'X_harmony' not in original_adata.obsm:
            raise ValueError(f"'{args.data_file}' does not contain 'X_harmony' in adata.obsm. "
                             "Cannot proceed with --prepare_harmony.")

        # Create a new AnnData object with X_harmony as the main .X
        adata_for_metaq = sc.AnnData(original_adata.obsm['X_harmony'])
        
        # Transfer obs (cell metadata) and ensure obs_names match
        adata_for_metaq.obs = original_adata.obs.copy()
        adata_for_metaq.obs_names = original_adata.obs_names

        # Assign dummy var_names for the Harmony dimensions
        adata_for_metaq.var_names = [f'harmony_dim_{i}' for i in range(adata_for_metaq.X.shape[1])]

        # Construct the new filename
        input_dir = os.path.dirname(args.data_file)
        input_filename_without_ext = os.path.splitext(os.path.basename(args.data_file))[0]
        harmony_temp_file_name = f"{input_filename_without_ext}_Xharmony.h5ad"
        
        # Set the path to the intermediate file
        harmony_temp_file_path = os.path.join(input_dir, harmony_temp_file_name)
        
        # Save the new AnnData object
        print(f"Saving prepared Harmony data to: {harmony_temp_file_path}")
        adata_for_metaq.write_h5ad(harmony_temp_file_path)

        # Update the data_file_for_metaq to point to this new file
        data_file_for_metaq = harmony_temp_file_path
        print("Harmony preparation complete.")

    # --- General Configuration and Output Path Construction ---
    # Extract filename from the data_file_for_metaq path for save_name
    filename_for_save_name = os.path.splitext(os.path.basename(data_file_for_metaq))[0]

    # --- REMOVED THE MANUAL METACELL NUMBER ADDITION ---
    save_name_formatted = f"{filename_for_save_name}" # <--- UPDATED: Removed _cells{args.metacell_num}


    print(f"\n--- MetaQ-sc Configuration ---")
    print(f"Data file (used for MetaQ): {data_file_for_metaq}")
    print(f"Requested Metacell Number: {args.metacell_num}")
    print(f"Data Type: {args.data_type}")
    print(f"Generated Save Name (excluding auto-added metacell_num): {save_name_formatted}") # Updated print statement
    print(f"Device: {args.device}")
    print(f"----------------------------")

    # --- Create output directories if they don't exist ---
    output_base_dir = "./save"
    figures_base_dir = "./figures"

    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(figures_base_dir, exist_ok=True)
    print(f"Created output directories: {output_base_dir}/ and {figures_base_dir}/")


    # --- Run MetaQ-sc ---
    run_metaq(
        data_path=[data_file_for_metaq],
        data_type=[args.data_type],
        metacell_num=args.metacell_num,
        save_name=save_name_formatted, # Use the cleaned save_name
        device=args.device,
    )
    print("\nMetaQ-sc script executed successfully!")

except Exception as e:
    print(f"\nError during MetaQ-sc execution: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Calculate and print total execution time
    if 'start_time' in locals():
        end_time = time.time()
        total_time_seconds = end_time - start_time
        hours, rem = divmod(total_time_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nTotal script execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    else:
        print("\nScript did not start properly, cannot calculate execution time.")
    
    # Clean up the intermediate Harmony file if it was created
    if harmony_temp_file_path and os.path.exists(harmony_temp_file_path):
        try:
            os.remove(harmony_temp_file_path)
            print(f"Removed temporary Harmony file: {harmony_temp_file_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary Harmony file {harmony_temp_file_path}: {e}")