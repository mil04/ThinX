import pytest
import numpy as np
import torch
from thinx.data_loader import DataLoader
from thinx.pytorch_nn import PyTorchNN
from xgboost import XGBClassifier, XGBRegressor
from thinx.tabular_preprocessor import TabularPreprocessor

# Use white_wine dataset (ID=44971) from OpenML
OPENML_DATASET_ID = 44971
OPENXAI_MODEL_NAME = "ann"

@pytest.fixture
def loader():
    return DataLoader()


def test_get_model_ann_classification(loader):
    model = loader._get_model("nn", task_type="classification", random_state=0)
    assert isinstance(model, PyTorchNN)


def test_get_model_xgboost_classification(loader):
    model = loader._get_model("xgboost", task_type="classification", random_state=0)
    assert isinstance(model, XGBClassifier)


def test_get_model_xgboost_regression(loader):
    model = loader._get_model("xgboost", task_type="regression", random_state=0)
    assert isinstance(model, XGBRegressor)


def test_get_model_invalid(loader):
    with pytest.raises(ValueError):
        loader._get_model("unsupported_model", task_type="classification", random_state=0)


@pytest.mark.slow
def test_load_from_openml_white_wine_ann(loader):
    dataset_name, X_train, y_train, X_test, y_test, model, preprocessor = loader.load_from_openml(
        dataset_id=OPENML_DATASET_ID,
        model_name="nn",
        task_type="regression"
    )
    assert "white_wine" in dataset_name.lower()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == X_test.shape[1]
    assert isinstance(model, PyTorchNN)
    assert isinstance(preprocessor, TabularPreprocessor)
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]


@pytest.mark.slow
def test_load_from_openml_white_wine_xgboost(loader):
    dataset_name, X_train, y_train, X_test, y_test, model, preprocessor = loader.load_from_openml(
        dataset_id=OPENML_DATASET_ID,
        model_name="xgboost",
        task_type="regression"
    )
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == X_test.shape[1]
    assert isinstance(model, XGBRegressor)
    assert isinstance(preprocessor, TabularPreprocessor)
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]


@pytest.mark.slow
def test_load_from_openml_task_inference(loader):
    dataset_name, X_train, y_train, X_test, y_test, model, preprocessor = loader.load_from_openml(
        dataset_id=OPENML_DATASET_ID,
        model_name="xgboost"
    )
    assert isinstance(model, XGBRegressor)

@pytest.mark.slow
def test_load_from_openxai(loader):
    X_test, y_test, model = loader.load_from_openxai(
        dataset_name="german",
        model_name=OPENXAI_MODEL_NAME
    )
    assert X_test.shape[0] > 0
    assert y_test.shape[0] == X_test.shape[0]
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(model, torch.nn.Module)


@pytest.mark.slow
def test_load_from_openxai_invalid_model(loader):
    with pytest.raises(ValueError):
        loader.load_from_openxai(dataset_name="german", model_name="xgboost")
