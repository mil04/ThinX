import numpy as np
import pandas as pd
import os
import sys
from goodpoints import compress
import thinx, thinx.utils
import argparse
import hashlib
import math

sys.stdout.reconfigure(line_buffering=True)

kernels = ["gaussian", "sobolev", "inverse_multiquadric", "matern"] # different kernels to evaluate
coefficients = [1/4, 1/2, 1, 2, 4] # compression coefficients to evaluate


def run_pipeline_with_kernels_and_iid(dataset_name, X_test, y_test, X_foreground, y_foreground, model, model_name, explainer_name, strategy, task_type, seed, n_jobs, n_repeats):
    N_REPEATS = n_repeats
    basic_size =int(math.sqrt(compress.largest_power_of_four(len(X_test)))) # base size for compression for COMPRESS++
    thinx.utils.set_global_seed(seed)

    # Load the ground truth explanations, calculated previously: mean over the runs
    gt = np.load(f"/mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/experiments/ground_truth/package_metadata/openml/{dataset_name}/ground_truth/{explainer_name}_{strategy}_3_{model_name}.npz")
    gt_exp_values, gt_times = gt["exp_values"], gt["times"]
    mean_gt_exp_values = np.mean(gt_exp_values, axis=0)

    evaluator = thinx.Evaluator(ground_truth_explanation=mean_gt_exp_values, ground_truth_points=X_test.copy())

    results = []

    print("\n[INFO] Running Kernel Thinning for different kernels...")
    for i in range(N_REPEATS):
        for kernel in kernels:
            for target_size in [basic_size * coeff for coeff in coefficients]:
                pre = thinx.Preprocessor(
                    X=X_test.copy(),
                    y=y_test.copy(),
                    model=model,
                    compression_method="kernel_thinning",
                    data_modification_method="none",
                    seed=seed + i + int(hashlib.sha256(kernel.encode()).hexdigest(), 16) % (10**6)
                )
                print(f"Repeat {i+1}/{N_REPEATS}, Kernel: {kernel}, Target Size: {target_size}")
                
                X_kt, y_kt, idx_kt, t_kt = pre.preprocess(
                    g=4,
                    num_bins=32,
                    target_size=target_size,                        
                    kernel=kernel
                )

                explainer = thinx.Explainer(
                    model=model,
                    task_type=task_type,
                    explainer_name=explainer_name,
                    strategy=strategy,
                    seed=int(seed + i + target_size + int(hashlib.sha256(kernel.encode()).hexdigest(), 16) % (10**6))
                )

                # calculate explanations using the compressed set as background
                exp_values, t_exp = explainer.explain(X_foreground=X_foreground, X_background=X_kt, y_foreground=y_foreground, n_jobs=n_jobs)

                row = evaluator.evaluate_explanation(exp_values, t_exp, len(X_kt))
                row.update(evaluator.evaluate_compression(X_kt))
                row.update({
                    "explainer": explainer_name,
                    "strategy": strategy,
                    "method": "kernel_thinning",
                    "kernel": kernel,
                    "g": 4,
                    "num_bins": 32,
                    "target_size": target_size,
                    "compression_time": t_kt,
                    "unique_samples": len(np.unique(idx_kt)),
                })
                results.append(row)

    print("\n[INFO] Running IID Baseline...")
    sizes = pd.DataFrame(results)["size"].unique()
    for i in sizes:
        for j in range(N_REPEATS):
            pre = thinx.Preprocessor(
                X=X_test.copy(),
                y=y_test.copy(),
                model=model,
                compression_method="iid",
                data_modification_method="none",
                seed=seed + j + i
            )
            X_iid, y_iid, idx_iid, t_iid = pre.preprocess(target_size=i)
            explainer = thinx.Explainer(
                model=model,
                task_type=task_type,
                explainer_name=explainer_name,
                strategy=strategy,
                seed=seed + j + i
            )
            exp_values, t_exp = explainer.explain(X_foreground=X_foreground, X_background=X_iid, y_foreground=y_foreground, n_jobs=n_jobs)
            
            row = evaluator.evaluate_explanation(exp_values, t_exp, len(X_iid))
            row.update(evaluator.evaluate_compression(X_iid))
            row.update({
                "explainer": explainer_name,
                "strategy": strategy,
                "method": "iid",
                "kernel": "Not needed",
                "g": "Not needed",
                "num_bins": "Not needed",
                "target_size": i,
                "compression_time": t_iid,
                "unique_samples": len(np.unique(idx_iid)),
            })
            results.append(row)

    df = pd.DataFrame(results)
    save_dir = f"/mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/experiments/ground_truth/package_metadata/openml/{dataset_name}/kernels_iid_comparison/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{explainer_name}_{strategy}_{model_name}_kernels_iid_{seed}_{N_REPEATS}.csv")

    df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--explainer_name", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, required=True)
    parser.add_argument("--n_repeats", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    thinx.utils.set_global_seed(0)

    loader = thinx.DataLoader()
    dataset_name, X_train, y_train, X_test, y_test, model, _ = loader.load_from_openml(
        dataset_id=int(args.dataset_id),
        model_name=args.model_name
    )

    print(f"\n[START] Preparing dataset: {dataset_name}")

    if args.model_name == "nn":
        model.model_.eval()

    # --- select fixed foreground points <= 4096 !!! the same as were used for ground truth explanations !!! ---
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
    run_pipeline_with_kernels_and_iid(
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
        seed=int(args.seed),
        n_jobs=int(args.n_jobs),
        n_repeats=int(args.n_repeats),
    )
