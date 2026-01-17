# ====================================================================
#  SLURM JOB SUBMISSION SCRIPT FOR CUSTOM KT METHODS WITH DATA 
#  MODIFICATIONS
#
#  - CUSTOM KT METHODS USING GAUSSIAN KERNEL:
#      * PREDICTIONS
#      * STRATIFIED (classification datasets only)
#
#  1. SMALL DATASETS (CC18_SMALL + CTR23_SMALL)
#       -> SHAP, SAGE, SHAP-IQ EXPLAINERS
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
#
# ====================================================================
import os
import thinx.utils

seed = 42

model_names = ["nn", "xgboost"]

explainers = [
    ("shap", "kernel", 16),
    ("sage", "permutation", 16),
    ("shapiq", "kernel", 16),
]

# explainers = [
#      ("expected_gradients", "na", 16),
# ]

compression_methods = ["kernel_thinning"]
data_modification_methods = ["predictions", "stratified"]

print("[START] Generating and submitting SLURM jobs for large datasets \n")

for data_modification_method in data_modification_methods:
    for compression_method in compression_methods:
        for model_name in model_names:
            # for evaluation of Expected Gradients explainer on large datasets
            # for dataset_id in CC18_LARGE+CTR23_LARGE: 
            for dataset_id in thinx.utils.CC18_SMALL+thinx.utils.CTR23_SMALL:
                dataset_name = f"{dataset_id}"
                for explainer_name, strategy, n_jobs in explainers:
                    
                    mem_gb = "100G"

                    if compression_method != "kernel_thinning":
                        raise ValueError("This script is only for kernel thinning compression method and its custom modifications.")
                    
                    if data_modification_method == "stratified" and dataset_id not in thinx.utils.CC18_LARGE+thinx.utils.CC18_SMALL:
                        print(f"--> Skipping dataset {dataset_name} for stratified method (this method is only for classification datasets).")
                        continue
                    
                    if data_modification_method=="predictions":
                        prefix = "p"
                    elif data_modification_method=="stratified":
                        prefix = "str"
                    else:
                        raise ValueError("Unknown data modification method.")

                    job_name = f"{prefix}_{dataset_name}_{explainer_name}_{model_name}_{compression_method}_{data_modification_method}"
                    sh_file = f"run_{job_name}.sh"

                    path_to_script = "/mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/experiments/scripts_other_baselines/predictions.py"
                    path_to_python = "/mnt/evafs/faculty/home/kbokhan/bsc_cte/bin/python"

                    with open(sh_file, "w") as f:

                        f.write(f"""#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p long
#SBATCH --time=120:00:00
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
--data_modification_method {data_modification_method} \\
--n_jobs {n_jobs} \\
--seed {seed} \\
""")

                    os.system(f"chmod +x {sh_file}")
                    os.system(f"sbatch {sh_file}")
                    print(f"--> Submitted job: {job_name}")



