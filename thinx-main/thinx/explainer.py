import time
import numpy as np
from typing import Optional, Tuple
import shap
import sys
import sage
import torch
import joblib
import captum
from thinx.pytorch_nn import PyTorchNN
import shapiq
from torch.utils.data import DataLoader, TensorDataset
from pydvl.influence.torch import CgInfluence
from thinx.utils import set_global_seed

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


class Explainer:
    """
    A class for calculating explanations: SHAP, SAGE, SHAP-IQ, Expected Gradients
    on X_foreground samples using the X_background samples.
    """
    def __init__(self, model, explainer_name: str, task_type: str, strategy: str = "na", seed: int = 0):
        """
        Initialize the Explainer object with a model and explanation settings.

        Args:
            model: A trained model object. For classification, it must implement `predict_proba`.
                For regression, it must implement `predict`.
            explainer_name (str): Name of the explainer to use.
                Options: {"shap", "sage", "shapiq", "expected_gradients"}
            task_type (str): Type of task.
                Options: {"classification", "regression"}
            strategy: Strategy for the explainer, if applicable.
                Options depend on the explainer (e.g., "kernel", "permutation", "na")
            seed (int): Random seed for reproducibility.

        Raises:
            - If `explainer_name` or `strategy` is not supported.
            - If the model does not have required methods for the specified `task_type`.
        """
        self.model = model
        self.explainer_name = explainer_name.lower()
        self.strategy = strategy.lower()
        self.task_type = task_type.lower()
        self.seed = seed

        if self.explainer_name not in {"shap", "sage", "shapiq", "expected_gradients", "influence"}:
            raise ValueError(f"Unsupported explainer: {self.explainer_name}")
        if self.strategy not in {"kernel", "permutation", "na"}:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
        
        if task_type == "classification" and hasattr(model, "predict_proba"):
            self.prediction_function = model.predict_proba
            self.loss = "cross entropy"
        elif task_type == "regression" and hasattr(model, "predict"):
            self.prediction_function = model.predict 
            self.loss = "mse"  
        else:
            raise ValueError("Model must have 'predict_proba' for classification or 'predict' for regression.")

    def explain(
        self,
        X_foreground: np.ndarray,
        X_background: np.ndarray,
        n_jobs: int = None,
        y_foreground: Optional[np.ndarray] = None,
        y_background: Optional[np.ndarray] = None, 
        verbose: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate explanations for the selected explaining method
        on X_foreground samples, using the X_background samples.

        Args:
            X_foreground (np.ndarray): Array of samples on which to calculate explanations.
            X_background (np.ndarray): Array of samples which are considered as representatives
                of the whole the distribution by explainer.
            n_jobs (int): Number of parallel jobs to use. If not specified (-1, or None),
                uses all available CPU cores.
            y_foreground (np.ndarray): Ground truth labels for the foreground samples.
            y_background (np.ndarray): Ground truth labels for the background samples.
            verbose (bool, optional): If True, prints progress messages. Default is True.

        Returns:
            Tuple[np.ndarray, float]:
                - `explanation_values`: Array of explanation values.
                - `time_elapsed`: Total time taken to compute the explanations.

        Raises:
            ValueError:
                - If `n_jobs` is invalid.
                - If required labels (`y_foreground` or `y_background`) are missing for SAGE or Influence explainers.
            RuntimeError:
                - If the explainer configuration is invalid or not supported.
        """
        if n_jobs is None or n_jobs == -1:
            n_jobs = joblib.cpu_count()
            print(f"Using all available CPU cores for parallel processing: {n_jobs} cores.")
        elif n_jobs > 0 and type(n_jobs) is int:
            print(f"Using {n_jobs} CPU cores for parallel processing.")
        else:
            raise ValueError("n_jobs must be None, -1, or a positive integer.")

        if self.explainer_name == "shap":
            return self._explain_shap(
                X_background=X_background,
                X_foreground=X_foreground,
                n_jobs=n_jobs,
                verbose=verbose
        )

        elif self.explainer_name == "sage":
            if y_foreground is None:
                raise ValueError("SAGE explanation requires labels (y).")
            return self._explain_sage(
                X_background=X_background,
                X_foreground=X_foreground,
                y_foreground=y_foreground,
                n_jobs=n_jobs,
                verbose=verbose
            )

        elif self.explainer_name == "expected_gradients":
            return self._explain_expected_gradients(
                X_background=X_background,
                X_foreground=X_foreground,
                n_jobs=n_jobs,
                verbose=verbose
            )

        elif self.explainer_name == "shapiq":
            return self._explain_shapiq(
                X_background=X_background,
                X_foreground=X_foreground,
                n_jobs=n_jobs,
                verbose=verbose
            )

        elif self.explainer_name == "influence":
            if y_foreground is None or y_background is None:
                raise ValueError("Influence explanation requires both foreground and background labels (y).")
            return self._explain_influence(
                X_background=X_background,
                y_background=y_background,
                X_foreground=X_foreground,
                y_foreground=y_foreground,
                n_jobs=n_jobs,
                verbose=verbose
            )

        raise RuntimeError("Invalid configuration.")

    def _explain_shap(
            self,
            X_background: np.ndarray, 
            X_foreground: np.ndarray, 
            n_jobs: int, 
            verbose: bool = True
        ) -> Tuple[np.ndarray, float]:
        """
        Calculates SHAP explanations.

        Args:
            X_background (np.ndarray): Array of samples which are considered as representatives
                of the whole the distribution.
            X_foreground (np.ndarray): Array of samples on which to calculate explanations.
            n_jobs: Number of parallel jobs to use. Must be >= 1.
                Parallelization is applied over batches of foreground samples.
            verbose (bool): Indicator, whether to print progress messages. Default is True.

        Returns:
            Tuple[np.ndarray, float]:
                - Computed SHAP values. Always of shape (n_samples, d)
                    Note: In case of classification: explanations are taken only
                    for the predicted class.
                - Total runtime in seconds, including initialization time and
                    per-batch explanation time.

        Raises:
            ValueError: If the SHAP strategy specified in the configuration is unknown.
        """
        print(
            f"Explaining with SHAP. {len(X_foreground)} samples to explain "
            f"using {len(X_background)} background samples."
        )

        # ------------------------- Initialization -------------------------
        start = time.time()
        if self.strategy == "kernel":
            explainer = shap.KernelExplainer(self.prediction_function, X_background, seed=self.seed)

        elif self.strategy == "permutation":
            masker = shap.maskers.Independent(X_background, max_samples=X_background.shape[0])
            explainer = shap.PermutationExplainer(self.prediction_function, masker, seed=self.seed)

        else:
            raise ValueError(f"Unknown strategy for SHAP: {self.strategy}")

        initialization_time = time.time() - start

        # ------------------------- Batching -------------------------
        BATCH_SIZE = 10
        batches = [
            X_foreground[(i*BATCH_SIZE):(i+1)*BATCH_SIZE] for i in range(int(1+X_foreground.shape[0]/BATCH_SIZE))
        ]

        if verbose:
            print(f"Running SHAP explanation in parallel using {n_jobs} jobs,"
                  f" {len(batches)} batches of size {BATCH_SIZE}")

        # ------------------------- Per-batch execution -------------------------
        def run_batch(batch: np.ndarray, batch_idx: int):
            set_global_seed(int(self.seed + batch_idx))
            nonlocal total_explanation_time
            t0 = time.time()
            shap_values_ = explainer(batch, silent=True).values
            batch_time = time.time() - t0

            if verbose:
                print(f"  → Finished batch {batch_idx + 1}/{len(batches)}")

            return shap_values_, batch_time

        # ------------------------- Parallel processing -------------------------
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(run_batch)(batch, idx)
            for idx, batch in enumerate(batches)
            if batch.shape[0] > 0
        )

        # ------------------------- Combine results -------------------------
        shap_values = np.concatenate([sv for sv, _ in results], axis=0)
        total_explanation_time = sum(batch_time for _, batch_time in results) + initialization_time

        if verbose:
            print(f"  → Finished all {X_foreground.shape[0]} samples")

        # ------------------------- Classification handling -------------------------
        if self.task_type == "classification":
            predictions = self.prediction_function(X_foreground)
            predicted_classes = np.argmax(predictions, axis=1)
            num_samples = X_foreground.shape[0]
            shap_values = shap_values[np.arange(num_samples), :, predicted_classes]

        return shap_values, total_explanation_time

    def _explain_sage(
            self,
            X_background: np.ndarray,
            X_foreground: np.ndarray,
            y_foreground: np.ndarray, 
            n_jobs: int, 
            verbose: bool = True
        ) -> Tuple[np.ndarray, float]:
        """
        Calculates SAGE explanations.

        Args:
            X_background (np.ndarray): Array of samples which are considered as representatives
                of the whole the distribution.
            X_foreground (np.ndarray): Array of samples on which to calculate explanations.
            y_foreground (np.ndarray): Array of target values for `X_foreground` samples.
            n_jobs: Number of parallel jobs to use. Must be >= 1.
            verbose (bool): Indicator, whether to print progress messages. Default is True.

        Returns:
            Tuple[np.ndarray, float]:
                - Computed SAGE explanations.
                - Total runtime in seconds.

        Raises:
            ValueError: If the SHAP strategy specified in the configuration is unknown.
        """
        print(f"Explaining with SAGE. {len(X_foreground)} samples to explain using {len(X_background)} background samples.")

        start = time.time()
        imputer = sage.MarginalImputer(self.prediction_function, X_background)

        # edge case
        if self.loss == "cross entropy":
            y_foreground = (y_foreground == 1).astype(int)
            if len(np.unique(y_foreground)) == 1:
                y_foreground[0] = 1 - y_foreground[0]

        if self.strategy == "kernel":
            explainer = sage.KernelEstimator(imputer, loss=self.loss, random_state=self.seed)
        elif self.strategy == "permutation":
            if n_jobs is None:
                explainer = sage.PermutationEstimator(imputer, loss=self.loss, random_state=self.seed)
            else:
                explainer = sage.PermutationEstimator(imputer, loss=self.loss, random_state=self.seed, n_jobs=n_jobs)
        else:
            raise ValueError(f"Unknown strategy for SAGE: {self.strategy}")
        
        sage_values = explainer(X_foreground, y_foreground, bar=False, verbose=verbose).values
        end = time.time()
        elapsed = end - start
        return sage_values, elapsed
    
    def _explain_shapiq(
            self,
            X_background: np.ndarray,
            X_foreground: np.ndarray,
            n_jobs: int, 
            verbose: bool = True
        ) -> Tuple[np.ndarray, float]:
        """
        Calculates SHAP-IQ explanations of order 2 (coalitions of size 2).

        Args:
            X_background (np.ndarray): Array of samples which are considered as representatives
                of the whole the distribution.
            X_foreground (np.ndarray): Array of samples on which to calculate explanations.
            n_jobs: Number of parallel jobs to use. Must be >= 1.
            verbose (bool): Indicator, whether to print progress messages. Default is True.

        Returns:
            Tuple[np.ndarray, float]:
                - Computed SHAP-IQ explanations for each sample in foreground samples.
                    Shape is (n_samples, d, d)
                - Total runtime in seconds spent on explanations.
        """
        print(f"Explaining with ShapIQ. {len(X_foreground)} samples to explain using {len(X_background)} background samples.")

        # ------------------------- Initialization -------------------------
        if self.task_type == "regression":
            model_func = self.prediction_function
        elif self.task_type == "classification":
            def model_func(X):
                proba = self.prediction_function(X)
                preds = np.argmax(proba, axis=1)
                return np.array([proba[i, preds[i]] for i in range(len(preds))])
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")

        imputer = shapiq.MarginalImputer(
            model=model_func,
            data=X_background,
            sample_size=len(X_background)
        )

        explainer = shapiq.TabularExplainer(
            model=model_func,
            data=X_background,
            approximator="regression",
            index="k-SII",
            max_order=2,
            imputer=imputer
        )

        # ------------------------- Per-sample execution -------------------------
        def explain_single(i):
            set_global_seed(int(self.seed + i))
            start = time.time()
            iv = explainer.explain(X_foreground[i], budget=4096, random_state=int(self.seed + i))
            main = np.asarray(iv.get_n_order_values(1)).ravel()
            try:
                pair = iv.get_n_order_values(2)
            except Exception:
                pair = None
            elapsed = time.time() - start

            if verbose:
                print(f"  → Finished sample {i + 1}/{X_foreground.shape[0]}")

            return main, pair, elapsed

        # ------------------------- Parallel processing -------------------------
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(explain_single)(int(i))
            for i in range(X_foreground.shape[0])
        )

        # ------------------------- Combine results -------------------------
        main_effects, pairwise_list, times = zip(*results)
        self.main_effects = np.vstack(main_effects)
        explanation_time = sum(times)

        return pairwise_list, explanation_time
    
    def _explain_expected_gradients(
            self,
            X_background: np.ndarray,
            X_foreground: np.ndarray,
            n_jobs: int = None, 
            verbose: bool = True
        ) -> Tuple[np.ndarray, float]:
        """
        Calculates Expected Gradients explanations.

        Args:
            X_background (np.ndarray): Array of samples which are considered as representatives
                of the whole the distribution.
            X_foreground (np.ndarray): Array of samples on which to calculate explanations.
            n_jobs: Number of parallel jobs to use. Must be >= 1.
            verbose (bool): Indicator, whether to print progress messages. Default is True.

        Returns:
            Tuple[np.ndarray, float]:
                - Computed Expected Gradients. Always of shape (n_samples, d)
                    Note: In case of classification: explanations are taken only
                    for the predicted class.
                - Total runtime in seconds, including initialization time and
                    per-batch explanation time.

        Raises:
            TypeError: If model is not a PyTorchNN instance.
            ValueError: If task_type is not 'classification' or 'regression'.
        """
        print(f"Explaining with Expected Gradients. {len(X_foreground)} samples to explain using {len(X_background)} background samples.")
        start = time.time()

        if not isinstance(self.model, PyTorchNN):
            raise TypeError("model must be an instance of PyTorchNN")

        inputs = torch.as_tensor(X_foreground, dtype=torch.float32)
        baselines = torch.as_tensor(X_background, dtype=torch.float32)
        explainer = captum.attr.IntegratedGradients(self.model.model_)

        explanations = []

        # ------------------------- Per-sample execution -------------------------
        def explain_sample(i, target_class=None):
            set_global_seed(int(self.seed + i))
            x_sample = inputs[i:i+1]

            tasks = []
            # --- run over all baselines ---
            for j in range(baselines.shape[0]):
                seed_j = self.seed + i * baselines.shape[0] + j
                set_global_seed(int(seed_j))
                if target_class is not None:
                    tasks.append(joblib.delayed(explainer.attribute)(x_sample, baselines[[j]], target=int(target_class)))
                else:
                    tasks.append(joblib.delayed(explainer.attribute)(x_sample, baselines[[j]]))

            # ------------------------- Parallel processing -------------------------
            results = joblib.Parallel(n_jobs=n_jobs)(tasks)
            explanation = torch.mean(torch.stack(results), dim=0)
            return explanation.detach().cpu().numpy().ravel()

        # ------------------------- Handle classification -------------------------
        if self.task_type == "classification":
            predictions = self.prediction_function(X_foreground)
            predicted_classes = np.argmax(predictions, axis=1)

            if verbose:
                print(f"Running Expected Gradients for {len(inputs)} samples (classification).")

            # --- run for all samples ---
            for i, target_class in enumerate(predicted_classes):
                explanations.append(explain_sample(i, target_class))
                if verbose:
                    print(f"  → Finished sample {i + 1}/{len(inputs)}")

        # ------------------------- Handle Regression -------------------------
        elif self.task_type == "regression":
            if verbose:
                print(f"Running Expected Gradients for {len(inputs)} samples (regression).")

            # --- run for all samples ---
            for i in range(inputs.shape[0]):
                explanations.append(explain_sample(int(i)))
                if verbose:
                    print(f"  → Finished sample {i + 1}/{len(inputs)}")

        else:
            raise ValueError("task_type must be 'classification' or 'regression'")

        final_explanations = torch.tensor(explanations, dtype=torch.float32)
        elapsed = time.time() - start

        if verbose:
            print(f"Done. Explained {len(inputs)} samples.")

        return final_explanations, elapsed

    def _explain_influence(
            self,
            X_background: np.ndarray,
            y_background: np.ndarray,
            X_foreground: np.ndarray,
            y_foreground: np.ndarray, 
            n_jobs: int = None,
            verbose: bool = True
        ) -> Tuple[np.ndarray, float]:
        """
        An experimental functions, which relies on influence functions, making explanations.

        Args:
            X_background (np.ndarray): Array of samples which are considered as representatives
                of the whole the distribution.
            y_background (np.ndarray): Ground truth labels for the background samples.
            X_foreground (np.ndarray): Array of samples on which to calculate explanations.
            y_foreground (np.ndarray): Ground truth labels for the foreground samples.
            n_jobs (int): Number of parallel jobs to use. If not specified (-1, or None),
                uses all available CPU cores. - NOT IMPLEMENTED
            verbose (bool, optional): If True, prints progress messages. Default is True.

        Returns:
            Tuple[np.ndarray, float]:
                - Compluted explanations.
                - Total runtime in seconds.
        """
        if verbose:
            print("Explaining using influence functions.")

        if isinstance(self.model, PyTorchNN):
            model = self.model.model_  
        elif isinstance(self.model, torch.nn.Module):
            model = self.model
        else:
            raise TypeError("Influence explainer requires a PyTorch model or PyTorchNN wrapper.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        start = time.time()

        X_train = torch.tensor(X_background).float().to(device)
        y_train = torch.tensor(y_background).long().to(device)

        X_test = torch.tensor(X_foreground).float().to(device)
        y_test = torch.tensor(y_foreground).long().to(device)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=False)

        if self.task_type == "classification":
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            loss_fn = torch.nn.MSELoss()

        influence_model = CgInfluence(
            model,
            loss_fn,
            regularization=1e-3,
            rtol=1e-7,
            atol=1e-7,
            solve_simultaneously=True
        ).fit(train_loader)

        influence_matrix = influence_model.influences(
            X_test, y_test, X_train, y_train, mode="up"
        ).cpu().numpy()

        elapsed = time.time() - start
        return influence_matrix, elapsed

