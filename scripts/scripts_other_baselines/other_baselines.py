import numpy as np
import pandas as pd
import os
import time
import sys
from goodpoints import compress
import thinx.utils, thinx
import argparse
import hashlib
import math

sys.stdout.reconfigure(line_buffering=True)

# target size coefficients for different baseline methods
coefficients_basic = [1/4, 1/2, 1, 2, 4]

def run_experiment_for_5_sizes(dataset_name, X_test, y_test, X_foreground, y_foreground, model, model_name, explainer_name, strategy, task_type, data_modification_method, compression_method, seed, n_jobs):
    print(f"[INFO] Running experiment for explainer: {explainer_name}, strategy: {strategy}, model: {model_name}, dataset: {dataset_name} - compression: {compression_method}, data modification: {data_modification_method}")
    coefficients = coefficients_basic

    N_REPEATS = 10 # for some datasets we had to do by one repeat only in one slurm job => 10 jobs in total for one dataset then
    basic_size =int(math.sqrt(compress.largest_power_of_four(len(X_test))))
    thinx.utils.set_global_seed(seed)

    gt = np.load(f"/mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/experiments/package_metadata/openml/{dataset_name}/ground_truth/{explainer_name}_{strategy}_3_{model_name}.npz")
    gt_exp_values, gt_times = gt["exp_values"], gt["times"]

    mean_gt_exp_values = np.mean(gt_exp_values, axis=0) # Average over repeats

    evaluator = thinx.Evaluator(ground_truth_explanation=mean_gt_exp_values, ground_truth_points=X_test.copy())

    results = []

    print(f"[INFO] Running {explainer_name}-{strategy} ({compression_method}, {data_modification_method}) on dataset: {dataset_name}")
    for i in range(N_REPEATS):
        for target_size in [basic_size * coeff for coeff in coefficients]:
            target_size = int(target_size)
            pre = thinx.Preprocessor(
                X=X_test.copy(),
                y=y_test.copy(),
                model=model,
                compression_method=compression_method,
                data_modification_method=data_modification_method,
                seed=int(seed + i + target_size)
            )
            print(f"Repeat {i+1}/{N_REPEATS}, target size: {target_size}")
            
            try:
                if compression_method == "stein_thinning":
                    X_comp, y_comp, idx_comp, t_comp = pre.preprocess(
                        target_size=target_size,
                        grad_type=b'gaussian'
                    )
                elif compression_method == "influence":
                    X_comp, y_comp, idx_comp, t_comp, _ = pre.preprocess(
                        target_size=target_size
                    )
                else:
                    X_comp, y_comp, idx_comp, t_comp = pre.preprocess(
                        target_size=target_size
                    )

            except Exception as e:
                # influence method can fail on a dataset due to numerical issues
                print(f"[ERROR] Compression failed for target_size={target_size} on repeat {i+1}: {e}")
                continue

            explainer = thinx.Explainer(
                model=model,
                task_type=task_type,
                explainer_name=explainer_name,
                strategy=strategy,
                seed=int(seed + i + target_size)
            )

            exp_values, t_exp = explainer.explain(X_foreground=X_foreground, X_background=X_comp, y_foreground=y_foreground, n_jobs=n_jobs)

            row = evaluator.evaluate_explanation(exp_values, t_exp, len(X_comp))
            row.update(evaluator.evaluate_compression(X_comp))
            row.update({
                "explainer": explainer_name,
                "strategy": strategy,
                "method": compression_method,
                "data_modification_method": data_modification_method,
                "kernel": "Not needed",
                "g": "Not needed",
                "num_bins": "Not needed",
                "target_size": target_size,
                "compression_time": t_comp,
                "unique_samples": len(np.unique(idx_comp)),
            })
            results.append(row)

    df = pd.DataFrame(results)
    save_dir = f"/mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/experiments/package_metadata/openml/{dataset_name}/other_baselines/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{compression_method}_{data_modification_method}_{explainer_name}_{strategy}_{model_name}_{seed}_{N_REPEATS}.csv")

    df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--explainer_name", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--compression_method", type=str, required=True)
    parser.add_argument("--data_modification_method", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if args.data_modification_method != "none" and args.compression_method != "kernel_thinning":
        raise ValueError("Data modification method can be used only with kernel thinning compression method.")

    thinx.utils.set_global_seed(0)

    loader = thinx.DataLoader()
    dataset_name, X_train, y_train, X_test, y_test, model, _ = loader.load_from_openml(
        dataset_id=int(args.dataset_id),
        model_name=args.model_name
    )

    print(f"\n[START] Preparing dataset: {dataset_name}")

    if args.model_name == "nn":
        model.model_.eval()

    # --- select fixed foreground points <= 4096 - !!! the same as in ground truth generation ---
    if (
        (args.explainer_name == "shap" and args.strategy == "kernel")
        or (args.explainer_name == "expected_gradients" and args.strategy == "na")
        or (args.explainer_name == "shapiq" and args.strategy == "kernel")
    ) and len(X_test) > 4096:
        rng_fg = np.random.default_rng(int(args.dataset_id))
        ids_fg = rng_fg.choice(len(X_test), size=4096, replace=False)
        X_foreground = X_test[ids_fg]
        y_foreground = y_test[ids_fg]
    else:
        X_foreground = X_test
        y_foreground = y_test
    # ---------------------------------------

    # determine task type
    if int(args.dataset_id) in thinx.utils.CC18_ALL:
        task_type = "classification"
    elif int(args.dataset_id) in thinx.utils.CTR23_ALL:
        task_type = "regression"
    else:
        raise ValueError(f"Dataset ID {args.dataset_id} not found in CC18 or CTR23 benchmarks.")

    print(f"[INFO] Starting {args.explainer_name}-{args.strategy} for dataset: {dataset_name}")
    run_experiment_for_5_sizes(
        dataset_name=dataset_name,
        X_test=X_test, 
        y_test=y_test,
        X_foreground=X_foreground,
        y_foreground=y_foreground,
        model=model,
        model_name=args.model_name,
        explainer_name=args.explainer_name,
        strategy=args.strategy,
        task_type=task_type,
        data_modification_method=args.data_modification_method,
        compression_method=args.compression_method,
        seed=int(args.seed),
        n_jobs=int(args.n_jobs)
    )
