import os
import sys
import numpy as np
import argparse
import thinx, thinx.utils

sys.stdout.reconfigure(line_buffering=True)


def run_explanations(
    dataset_name,
    X_test,
    y_test,
    X_foreground,
    y_foreground,
    model,
    model_name,
    explainer_name,
    strategy,
    task_type,
    num_repeats,
    seed,
    n_jobs,
):
    print(f"\n=== Running {explainer_name} ({strategy}) for dataset: {dataset_name} ===")

    explanations = []
    times = []

    for i in range(num_repeats):
        print(f"\n[INFO] Repeat {i + 1}")

        # --- resample background points each repeat, if there are more than 5000 test samples ---
        rng_bg = np.random.default_rng(seed + i)
        if len(X_test) > 5000:
            ids_bg = rng_bg.choice(len(X_test), size=5000, replace=False)
            X_background = X_test[ids_bg]
        else:
            X_background = X_test
        # ------------------------------------------------

        explainer = thinx.Explainer(
            model=model,
            explainer_name=explainer_name,
            strategy=strategy,
            task_type=task_type,
            seed=seed + i
        )

        exp, elapsed_time = explainer.explain(
            X_background=X_background,
            X_foreground=X_foreground,
            y_foreground=y_foreground,
            n_jobs=n_jobs,
        )

        explanations.append(exp)
        times.append(elapsed_time)

    explanations = np.array(explanations)
    times = np.array(times)

    save_dir = f"package_metadata/openml/{dataset_name}/ground_truth"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{explainer_name}_{strategy}_{num_repeats}_{model_name}.npz")

    np.savez_compressed(save_path, exp_values=explanations, times=times)

    print(f"[DONE] Saved explanations and timings to: {save_path}")
    print(f"[DONE] Completed {num_repeats} repeats for {dataset_name}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--explainer_name", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, required=True)
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

    # --- select fixed foreground points <= 4096 samples ---
    if (
        (args.explainer_name == "shap" and args.strategy == "kernel")
        or (args.explainer_name == "expected_gradients" and args.strategy == "na")
        or (args.explainer_name == "shapiq" and args.strategy == "kernel")
    ) and len(X_test) > 4096:
        rng_fg = np.random.default_rng(int(args.dataset_id))
        ids_fg = rng_fg.choice(len(X_test), size=4096, replace=False)
        X_foreground = X_test[ids_fg]
        y_foreground = y_test[ids_fg]
    else: # case of SAGE Permutation or small test set
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

    run_explanations(
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
        num_repeats=3,
        seed=42,
        n_jobs=int(args.n_jobs),
    )
