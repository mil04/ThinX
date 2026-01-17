# =====================================================================
#  SLURM JOB SUBMISSION SCRIPT — SMALL DATASETS (CC18 + CTR23)
#  SHAP & SAGE EXPLAINERS ON NN AND XGBOOST MODELS - 
#  KERNEL THINNING WITH 5 DIFFERENT COMPRESSION SIZES 
#  AND WITH KERNELS: Gaussian, Inverse Multiquadric, Matérn, Sobolev
#  IID SAMPLING FOR THESE SIZES IS ALSO CALCULATED
# =====================================================================
import os
from thinx.utils import CC18_SMALL, CTR23_SMALL

seed = 42
n_repeats = 10

model_names = ["nn", "xgboost"]

explainers = [
    ("shap", "kernel", 16),
    ("sage", "permutation", 16),
    ("shapiq", "kernel", 16)
]

print("[START] Generating and submitting SLURM jobs for small datasets \n")

for model_name in model_names:
    for dataset_id in CC18_SMALL + CTR23_SMALL:
        dataset_name = f"{dataset_id}"
        for explainer_name, strategy, n_jobs in explainers:
            mem_gb = "100G"

            job_name = f"k_{dataset_name}_{explainer_name}_{model_name}"
            sh_file = f"run_{job_name}.sh"

            path_to_script = "/mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/experiments/scripts_kernels/kernels_iid.py"
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
  --n_jobs {n_jobs} \\
  --n_repeats {n_repeats} \\
  --seed {seed} 
""")

            os.system(f"chmod +x {sh_file}")
            os.system(f"sbatch {sh_file}")
            print(f"--> Submitted job: {job_name}")



# =====================================================================
#  SLURM JOB SUBMISSION SCRIPT — LARGE DATASETS (CC18 + CTR23)
#  EXPECTED GRADIENTS EXPLAINER ON NN MODEL - 
#  KERNEL THINNING WITH 5 DIFFERENT COMPRESSION SIZES 
#  AND WITH KERNELS: Gaussian, Inverse Multiquadric, Matérn, Sobolev
#  IID SAMPLING FOR THESE SIZES IS ALSO CALCULATED
# =====================================================================
import os
from thinx.core.utils import CC18_LARGE, CTR23_LARGE

model_name = "nn"

explainers = [
    ("expected_gradients", "na", 16),
]

print("\n[START] Generating and submitting SLURM jobs for large datasets \n")

for dataset_id in CC18_LARGE + CTR23_LARGE:
    dataset_name = f"{dataset_id}"
    for explainer_name, strategy, n_jobs in explainers:
        mem_gb = "100G"

        job_name = f"k_{dataset_name}_{explainer_name}_{model_name}"
        sh_file = f"run_{job_name}.sh"

        path_to_script = "/mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/experiments/scripts_kernels/kernels_iid.py"
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
  --n_jobs {n_jobs} \\
  --n_repeats {n_repeats} \\
  --seed {seed} 
""")

        os.system(f"chmod +x {sh_file}")
        os.system(f"sbatch {sh_file}")
        print(f"--> Submitted job: {job_name}")