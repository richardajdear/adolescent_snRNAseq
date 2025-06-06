#!/bin/bash
#SBATCH -A VERTES-SL2-GPU
#SBATCH -p ampere
#SBATCH -J metaq                            # Job name
#SBATCH --nodes=1                           # Request 1 node
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --time=02:00:00                     # Set a maximum run time (HH:MM:SS)
#SBATCH --output=metaq_job_%j.out           # Standard output and error log (%j expands to jobID)
#SBATCH --error=metaq_job_%j.err            # Standard error log


singularity exec --nv \
                 --cleanenv \
                 metaq.sif \
                 micromamba run -n metaq_env \
                 python ../code/run_metaq.py