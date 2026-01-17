import numpy as np
from typing import Dict
from thinx.metrics import compute_mae, compute_mmd, top_k_score, topk_pair_overlap

class Evaluator:
    """
    Evaluates quality of explanations against `ground_truth` using MAE, top-k, MMD, explanation_time.
    """
    def __init__(self, ground_truth_explanation: np.ndarray, ground_truth_points: np.ndarray):
        self.ground_truth_explanation = np.asarray(ground_truth_explanation)
        self.ground_truth_points = np.asarray(ground_truth_points)

    @staticmethod
    def _ensure_finite(arr: np.ndarray, name: str):
        """
        Validates that all values in the array are finite and not None.

        Args:
            arr (np.ndarray): Array to validate.
            name (str): Human-readable name used in error messages.

        Raises:
            ValueError: If the array contains NaN or infinite values.
        """
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN/Inf")

    def evaluate_explanation(
        self,
        explanation: np.ndarray,
        time_elapsed: float,
        num_samples: int
    ) -> Dict[str, float]:
        """
        Evaluates an explanation against the `ground_truth`.

        Args:
            explanation (np.ndarray): Explanation to evaluate.
            time_elapsed (float): Time used to compute this explanation.
            num_samples (int): Number of samples used to compute this explanation.

        Returns:
            Dict[str, float]: MAE, top-k score, explanation time, sample count.

        Raises:
            ValueError: On invalid shapes, types, or non-finite values.
            ValueError: If ndim of explanation is not 1, 2, 3.
        """
        explanation = np.asarray(explanation)

        if explanation.shape != self.ground_truth_explanation.shape:
            raise ValueError(
                f"explanation shape {explanation.shape} "
                f"!= ground truth {self.ground_truth_explanation.shape}"
            )

        if time_elapsed < 0:
            raise ValueError("time_elapsed must be non-negative")
        if not isinstance(num_samples, (int, np.integer)) or num_samples < 0:
            raise ValueError("num_samples must be a non-negative integer")

        self._ensure_finite(explanation, "explanation")
        self._ensure_finite(self.ground_truth_explanation, "ground truth explanation")

        mae = compute_mae(explanation, self.ground_truth_explanation)
        k = max(1, min(5, explanation.shape[-1]))
        if explanation.ndim == 1 or explanation.ndim == 2: # SHAP or SAGE od Expected Gradients case
            top_k = top_k_score(explanation, self.ground_truth_explanation, k=k)
        elif explanation.ndim == 3: # SHAP-IQ case
            top_k = topk_pair_overlap(explanation, self.ground_truth_explanation, k=k)
        else:
            raise ValueError("explanation must be 1D or 2D")

        return {
            "mae": float(mae),
            "top_k": float(top_k),
            "explanation_time": float(time_elapsed),
            "size": int(num_samples),
        }

    def evaluate_compression(self, compressed_points: np.ndarray) -> Dict[str, float]:
        """
        Evaluates quality of compressed points using MMD (Maximum Mean Discrepancy) metric.

        Args:
            compressed_points (np.ndarray): set of points achieved by compression procedure - 2D array.

        Returns:
            Dict[str, float]: MMD value.

        Raises:
            ValueError: On invalid shapes or non-finite values.
        """
        compressed_points = np.asarray(compressed_points)

        if compressed_points.ndim != 2 or self.ground_truth_points.ndim != 2:
            raise ValueError("points must be 2D arrays")
        if compressed_points.shape[1] != self.ground_truth_points.shape[1]:
            raise ValueError(
                f"Dim mismatch: compressed has {compressed_points.shape[1]} features, "
                f"ground truth has {self.ground_truth_points.shape[1]}"
            )
        self._ensure_finite(compressed_points, "compressed points")
        self._ensure_finite(self.ground_truth_points, "ground truth points")

        mmd = compute_mmd(self.ground_truth_points, compressed_points)
        return {"mmd": float(mmd)}
