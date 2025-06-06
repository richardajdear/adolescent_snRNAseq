# my_metaq_script.py
import torch
import numpy
import MetaQ_sc

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
print(f"NumPy version: {numpy.__version__}")
print(f"MetaQ-sc installed: {'MetaQ_sc' in globals()}") # Simple check if module exists

from MetaQ_sc import run_metaq

try:
    run_metaq(
        data_path=["/home/rajd2/rds/hpc-work/snRNAseq/data/velmeshev100k.h5ad"],
        data_type=["RNA"],
        metacell_num=1000,
        save_name="vel100k_cells1000",
        device="cuda"
    )
    print("MetaQ-sc script executed successfully!")
except Exception as e:
    print(f"Error during MetaQ-sc execution: {e}")