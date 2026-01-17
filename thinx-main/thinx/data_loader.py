import numpy as np
import openml
import torch
from typing import Union, Tuple, Optional
from thinx.pytorch_nn import PyTorchNN
from thinx.tabular_preprocessor import TabularPreprocessor
from thinx.utils import CC18_ALL, CTR23_ALL
from numpy import ndarray
from openxai.model import LoadModel, ReturnLoaders
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor


class DataLoader:
    """
    A class to handle loading of preprocessed data and trained models from various data sources like OpenML and OpenXAI.
    """

    def __init__(self):
        pass

    def load_from_openxai(
        self, dataset_name: str, model_name: str
    ) -> tuple[np.ndarray, np.ndarray, torch.nn.Module]:
        """
        Load test data and a pretrained model from OpenXAI.

        Args:
            dataset_name (str): Dataset name, e.g., "german".
            model_name (str): Model name. Only 'nn' is supported.

        Returns:
            Tuple[np.ndarray, np.ndarray, torch.nn.Module]:
                - X_test: Test samples.
                - y_test: Test target values.
                - model: Pretrained  NN from OpenXAI.

        Raises:
            ValueError: If model_name is not 'nn'.
        """
        if model_name != "ann":
            raise ValueError(f"Model '{model_name}' is not supported. Only 'nn' is available.")

        # --- Load data ---
        _, loader_test = ReturnLoaders(data_name=dataset_name, download=True, batch_size=128)
        X_test = np.asarray(loader_test.dataset.data)
        y_test = np.asarray(loader_test.dataset.targets)

        # --- Load model ---
        model = LoadModel(data_name=dataset_name, ml_model=model_name, pretrained=True)
        model.eval()

        return X_test, y_test, model

    def _get_model(
            self, model_name: str, task_type: str, random_state: int
    ) -> Union[PyTorchNN, XGBClassifier, XGBRegressor]:
        """
        Initialize a model based on its name and task type - used for datasets from OpenML.

        Args:
            model_name (str): 'nn' or 'xgboost'.
            task_type (str): 'classification' or 'regression'.
            random_state (int): Random seed for reproducibility.

        Returns:
            Initialized model object.

        Raises:
            ValueError: If model_name is not supported.
        """
        model_map = {
            "nn": PyTorchNN(task_type=task_type, random_state=0),
            "xgboost": (
                XGBClassifier(n_estimators=200, random_state=random_state)
                if task_type == "classification"
                else XGBRegressor(n_estimators=200, random_state=random_state)
            ),
        }
        model = model_map.get(model_name)
        if model is None:
            raise ValueError(
                f"Model '{model_name}' is not supported. "
                f"Choose from {list(model_map.keys())}."
            )
        return model

    def load_from_openml(
        self, dataset_id: int, model_name: str, task_type: Optional[str] = None
    ) -> Tuple[
        str,
        ndarray,
        ndarray,
        ndarray,
        Optional[ndarray],
        Union[PyTorchNN, XGBClassifier, XGBRegressor],
        TabularPreprocessor
    ]:
        """
        Loads data from OpenML, preprocesses it, trains a model, and returns all components.

        Args:
            dataset_id (int): OpenML dataset ID.
            model_name (str): 'nn' or 'xgboost'.
            task_type (Optional[str]): 'classification' or 'regression'.
                If None, inferred from dataset ID.

        Returns:
            Tuple containing:
                - dataset_name (str)
                - X_train (ndarray): Preprocessed training features.
                - y_train (ndarray): Training target.
                - X_test (ndarray): Preprocessed test features.
                - y_test (ndarray or None): Test target.
                - model: Trained model object.
                - preprocessor: Fitted TabularPreprocessor instance.

        Raises:
            ValueError: If task_type cannot be inferred.
        """
        # --- Infer task type if not provided ---
        if task_type is None:
            if dataset_id in CC18_ALL:
                task_type = "classification"
            elif dataset_id in CTR23_ALL:
                task_type = "regression"
            else:
                raise ValueError(
                    f"Task type for dataset ID {dataset_id} is unknown. "
                    "Please specify `task_type` ('classification' or 'regression')."
                )

        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

        # The `random_state` is fixed to ensure the same split across experiments.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=dataset_id
        )

        preprocessor = TabularPreprocessor(
            task_type=task_type,
            id_like_threshold=0.99,
            scale_all_numeric_after_encoding=True,
            scale_target_in_regression=True,
            random_state=0, # ensures reproducible preprocessing over experiments we perform.
        )
        # --- Preprocess data ---
        X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)

        # --- Train the model ---
        model = self._get_model(model_name, task_type, random_state=dataset_id)
        model.fit(X_train_processed.values, y_train_processed)

        return (
            dataset.name,
            X_train_processed.to_numpy(),
            y_train_processed,
            X_test_processed.to_numpy(),
            y_test_processed,
            model,
            preprocessor,
        )