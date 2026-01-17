# ====================================================================
#  SLURM JOB SUBMISSION SCRIPT FOR EXPERIMENTS WITH DIFFERENT BASELINES 
#  WITH DIFFERENT SIZES 
#  
#  - STEIN THINNING
#  - ARF COMPRESSION
#  - INFLUENCE-BASED COMPRESSION (only for NN models)
# 
#  1. SMALL DATASETS (CC18_SMALL + CTR23_SMALL) 
#       -> SHAP & SAGE & SHAP-IQ EXPLAINERS 
#       -> NN AND XGBOOST MODELS 
# 
#  2. LARGE DATASETS (CC18_LARGE + CTR23_LARGE)
#       -> EXPECTED GRADIENTS EXPLAINER
#       -> NN MODELS ONLY
# 
#  NOTE:
#  The current configuration runs experiments on SMALL datasets.
#  To switch to LARGE datasets, replace the dataset list and
#  the explainer with Expected Gradients.
# ====================================================================
import os
from thinx.utils import CC18_SMALL, CTR23_SMALL

seed = 42

# set a model
model_names = ["nn", "xgboost"]

explainers = [
    ("shap", "kernel", 16),
    ("sage", "permutation", 16),
    ("shapiq", "kernel", 16),
]

# In case we test on LARGE datasets and we want to run only expected gradients
# explainers = [
#     ("expected_gradients", "na", 16)
# ]

compression_methods = ["arfpy", "stein_thinning", "influence"] # influence works only for neural networks
data_modification_method = ["none"]

print("[START] Generating and submitting SLURM jobs for small datasets \n")

for data_modification in data_modification_method:
    for compression_method in compression_methods:
        for model_name in model_names:
            # change to CC18_LARGE + CTR23_LARGE to run on large datasets
            for dataset_id in CC18_SMALL + CTR23_SMALL:
                dataset_name = f"{dataset_id}"
                for explainer_name, strategy, n_jobs in explainers:
                    if compression_method == "influence" and model_name != "nn":
                        continue
                    
                    mem_gb = "60G"

                    if compression_method == "influence":
                        prefix = "i"
                    elif compression_method == "stein_thinning":
                        prefix = "st"
                    elif compression_method == "arfpy":
                        prefix = "a"

                    job_name = f"{prefix}_{dataset_name}_{explainer_name}_{model_name}_{compression_method}_{data_modification}"
                    sh_file = f"run_{job_name}.sh"

                    path_to_script = "/mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/experiments/scripts_other_baselines/other_baselines.py"
                    path_to_python = "/mnt/evafs/faculty/home/kbokhan/bsc_cte/bin/python"

                    with open(sh_file, "w") as f:

                        f.write(f"""#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p long
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem={mem_gb}
#SBATCH --nodelist=dgx-1,dgx-2,dgx-3,dgx-4
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_log.txt
#SBATCH --error=logs/{job_name}_err.txt
#SBATCH --cpus-per-task={n_jobs}
#SBATCH --gres=gpu:0
#SBATCH --mail-user=kateqwerty001@gmail.com
#SBATCH --mail-type=ALL

# Run the Python script with all arguments
{path_to_python} {path_to_script} \\
--dataset_id {dataset_id} \\
--model_name {model_name} \\
--explainer_name {explainer_name} \\
--strategy {strategy} \\
--compression_method {compression_method} \\
--data_modification_method {data_modification} \\
--n_jobs {n_jobs} \\
--seed {seed} \\
""")

                    os.system(f"chmod +x {sh_file}")
                    os.system(f"sbatch {sh_file}")
                    print(f"--> Submitted job: {job_name}")