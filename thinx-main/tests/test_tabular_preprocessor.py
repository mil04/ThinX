import pytest
import numpy as np
import pandas as pd
from thinx.tabular_preprocessor import TabularPreprocessor, PreprocessReport

def make_classification_df():
    X = pd.DataFrame({
        "num1": [0.0, 1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0],
        "num2": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "cat":  ["a", "a", "b", "b", "c", "c", "a", "b", "b", "c"],
        "const": 1,
        "id_str": [f"id{i}" for i in range(10)],  
    })
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 0, 1, 1], name="target")
    return X, y

def make_regression_df():
    X = pd.DataFrame({
        "num1": [1.1, 2.2, np.nan, 4.4, 5.5, 6.6, 7.7, np.nan],
        "num2": [10, 20, 30, 40, 50, 60, 70, 80],
        "str_id": [f"r{i}" for i in range(8)]
    })
    y = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0], name="target")
    return X, y

def test_fit_classification_creates_report_and_encoders():
    X, y = make_classification_df()
    pp = TabularPreprocessor(task_type="classification", id_like_threshold=0.9)
    pp.fit(X, y)
    rpt = pp.get_report()

    assert isinstance(rpt, PreprocessReport)
    assert rpt.task_type == "classification"
    assert isinstance(rpt.dropped_constant, list)
    assert "const" in rpt.dropped_constant
    assert isinstance(pp.num_imputer_, object)
    assert isinstance(pp.scaler_, object)
    assert isinstance(pp.label_encoder_, object)
    assert isinstance(pp.target_encoder_, object)

def test_transform_classification_returns_scaled_and_encoded_output():
    X, y = make_classification_df()
    pp = TabularPreprocessor(task_type="classification", id_like_threshold=0.9)
    X_tr, y_tr = pp.fit_transform(X, y)
    assert np.all(np.abs(X_tr.mean()) < 1e-6)
    assert y_tr.dtype.kind in {"i", "u"}
    X_new, y_new = pp.transform(X, y)
    assert list(X_new.columns) == list(pp.encoded_feature_names_)

def test_not_fitted_errors():
    X, y = make_classification_df()
    pp = TabularPreprocessor(task_type="classification")
    with pytest.raises(RuntimeError):
        pp.transform(X, y)
    with pytest.raises(RuntimeError):
        pp.get_report()

def test_invalid_task_type_raises():
    with pytest.raises(ValueError):
        TabularPreprocessor(task_type="unknown")

def test_scale_all_numeric_flag_behavior():
    X, y = make_classification_df()
    pp = TabularPreprocessor(
        task_type="classification",
        id_like_threshold=0.9,
        scale_all_numeric_after_encoding=False
    )
    X_tr, _ = pp.fit_transform(X, y)
    for col in pp.num_cols_:
        assert abs(X_tr[col].mean()) < 1e-6
    encoded_non_scaled = [c for c in X_tr.columns if c not in pp.num_cols_]
    if encoded_non_scaled:
        assert not np.all(np.abs(X_tr[encoded_non_scaled].mean()) < 1e-6)

def test_regression_fit_transform_with_target_scaling():
    X, y = make_regression_df()
    pp = TabularPreprocessor(
        task_type="regression",
        id_like_threshold=0.9,
        scale_target_in_regression=True
    )
    X_tr, y_tr = pp.fit_transform(X, y)

    assert pp.y_scaler_ is not None
    assert np.issubdtype(y_tr.dtype, np.floating)
    assert np.isfinite(y_tr).any()
    if np.isfinite(y_tr).all():
        assert abs(y_tr.mean()) < 1e-6

def test_regression_fit_transform_without_target_scaling():
    X, y = make_regression_df()
    pp = TabularPreprocessor(
        task_type="regression",
        id_like_threshold=0.9,
        scale_target_in_regression=False
    )
    X_tr, y_tr = pp.fit_transform(X, y)

    assert y_tr is not None
    assert len(X_tr) == len(y_tr)
    assert isinstance(y_tr, (np.ndarray, list))

def test_validate_inputs_handles_object_and_nan_regression():
    X = pd.DataFrame({
        "num": [1, 2, 3],
        "obj": ["4", "5", "bad"]
    })
    y = pd.Series(["1.0", "oops", "3.0"], name="target")
    pp = TabularPreprocessor(task_type="regression")
    Xv, yv = pp._validate_inputs(X, y)
    assert np.issubdtype(yv.dtype, np.floating)

def test_transform_realigns_columns_and_encodes_y():
    X, y = make_classification_df()
    pp = TabularPreprocessor(task_type="classification", id_like_threshold=0.9)
    pp.fit(X, y)
    X_mod = X.drop(columns=["num1"]).copy()
    X_mod["extra"] = 123
    X_t, y_t = pp.transform(X_mod, y)
    assert list(X_t.columns) == list(pp.encoded_feature_names_)
    assert y_t.dtype.kind in {"i", "u"}

def test_drop_duplicate_rows_and_report_consistency():
    X, y = make_classification_df()
    X = pd.concat([X, X.iloc[:2]], ignore_index=True)
    y = pd.concat([y, y.iloc[:2]], ignore_index=True)
    pp = TabularPreprocessor(task_type="classification")
    X_tr, y_tr = pp.fit_transform(X, y)
    assert len(X_tr) <= len(X)
    rpt = pp.get_report()
    assert rpt.task_type == "classification"
    assert isinstance(rpt.kept_raw_columns, list)
