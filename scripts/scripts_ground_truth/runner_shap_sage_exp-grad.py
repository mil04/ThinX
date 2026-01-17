# ============================================================
#  SLURM JOB SUBMISSION SCRIPT — SMALL DATASETS (CC18 + CTR23)
#  SHAP & SAGE & SHAP-IQ EXPLAINERS ON NN AND XGBOOST MODELS 
#  [GROUND TRUTH]
# ============================================================
import os
from thinx.utils import CC18_SMALL, CTR23_SMALL

model_names = ["nn", "xgboost"]

explainers = [
    ("shap", "kernel", 16),
    ("sage", "permutation", 16),
    ("shapiq", "kernel", 16),
]

print("[START] Generating and submitting SLURM jobs for small datasets \n")

for model_name in model_names:
    for dataset_id in CC18_SMALL + CTR23_SMALL:
        dataset_name = f"{dataset_id}"
        for explainer_name, strategy, n_jobs in explainers:
            
            # some datasets require more memory
            if explainer_name == "sage":
                mem_gb = "120G"
            elif explainer_name == "shap" or explainer_name == "shapiq":
                mem_gb = "100G"

            job_name = f"{dataset_name}_{explainer_name}_{model_name}"
            sh_file = f"run_{job_name}.sh"

            path_to_script = "/mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/experiments/ground_truth/gt_shap_sage_exp-grad.py"
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

# Run the Python script with all arguments
{path_to_python} {path_to_script} \\
  --dataset_id {dataset_id} \\
  --model_name {model_name} \\
  --explainer_name {explainer_name} \\
  --strategy {strategy} \\
  --n_jobs {n_jobs}
""")

            os.system(f"chmod +x {sh_file}")
            os.system(f"sbatch {sh_file}")
            print(f"--> Submitted job: {job_name}")




# ============================================================
#  SLURM JOB SUBMISSION SCRIPT — LARGE DATASETS (CC18 + CTR23)
#  EXPECTED GRADIENTS ON NN MODELS [GROUND TRUTH]
# ============================================================
import os
from thinx.utils import CC18_LARGE, CTR23_LARGE

model_name = "nn"

explainers = [
    ("expected_gradients", "na", 16),
]

print("\n[START] Generating and submitting SLURM jobs for large datasets \n")

for dataset_id in CC18_LARGE + CTR23_LARGE:
    dataset_name = f"{dataset_id}"
    for explainer_name, strategy, n_jobs in explainers:

        if explainer_name == "expected_gradients":
            mem_gb = "100G"

        job_name = f"{dataset_name}_{explainer_name}_{model_name}"
        sh_file = f"run_{job_name}.sh"

        path_to_script = "/mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/experiments/ground_truth/gt_shap_sage_exp-grad.py"
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

# Run the Python script with all arguments
{path_to_python} {path_to_script} \\
  --dataset_id {dataset_id} \\
  --model_name {model_name} \\
  --explainer_name {explainer_name} \\
  --strategy {strategy} \\
  --n_jobs {n_jobs}
""")

        os.system(f"chmod +x {sh_file}")
        os.system(f"sbatch {sh_file}")
        print(f"--> Submitted job: {job_name}")


