import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, TargetEncoder
from sklearn.impute import SimpleImputer


@dataclass
class PreprocessReport:
    """
    Container for storing information about performed preprocessing steps.
    """
    dropped_constant: List[str]
    dropped_id_like: List[str]
    kept_raw_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    task_type: str


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """
    A preprocessing class for tabular data. Works on any pandas DataFrame X and 1D target y.

    Steps:
      1) Drop duplicate rows (based on [X|y])
      2) Drop constant columns
      3) Drop ID-like columns (nunique / n_rows >= id_like_threshold)
      4) Impute: numeric(mean), categorical(most_frequent)
      5) Encode categoricals with TargetEncoder 
      6) Scale numeric features with StandardScaler
      7) Encode target: LabelEncoder for classification; (optional) standardize y for regression

    After fit, transform() applies the same columns/encoders/imputers/scalers to new data.
    """

    def __init__(
        self,
        task_type: str,  # 'classification' or 'regression'
        id_like_threshold: float = 0.99,
        scale_all_numeric_after_encoding: bool = True,
        scale_target_in_regression: bool = False,
        random_state: int = 0,
    ):
        """
        Initialize the preprocessing pipeline.

        Args:
            task_type (str): Either 'classification' or 'regression'.
            id_like_threshold (float): Uniqueness ratio threshold for identifying ID-like columns.
            scale_all_numeric_after_encoding (bool): Whether to scale numeric features after encoding.
            scale_target_in_regression (bool): Whether to scale the target in regression tasks.
            random_state (int): Seed for reproducibility.
        """
        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'")
        self.task_type = task_type
        self.id_like_threshold = id_like_threshold
        self.scale_all_numeric_after_encoding = scale_all_numeric_after_encoding
        self.scale_target_in_regression = scale_target_in_regression
        self.random_state = random_state

        # set during fit
        self.report_: Optional[PreprocessReport] = None
        self.raw_cols_: Optional[List[str]] = None
        self.num_cols_: Optional[List[str]] = None
        self.cat_cols_: Optional[List[str]] = None
        self.keep_cols_: Optional[List[str]] = None
        self.constant_cols_: Optional[List[str]] = None
        self.id_like_cols_: Optional[List[str]] = None
        self.num_imputer_: Optional[SimpleImputer] = None
        self.cat_imputer_: Optional[SimpleImputer] = None
        self.target_encoder_: Optional[TargetEncoder] = None
        self.scaler_: Optional[StandardScaler] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.y_scaler_: Optional[StandardScaler] = None
        self.task_type_: Optional[str] = None
        self.encoded_feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the preprocessing pipeline on the given data.
        Does not transform the data.

        Args:
            X (pd.DataFrame): Dataframe to fit.
            y (pd.Series): Target variables.

        Returns:
            self
        """
        X, y = self._validate_inputs(X, y)

        # drop ID-like columns (recompute after dedup)
        nunique = X.nunique(dropna=False)
        n_rows = len(X)
        candidate_cols = nunique[(nunique > 1) & (nunique >= n_rows * self.id_like_threshold)].index
        self.id_like_cols_ = [col for col in candidate_cols if not pd.api.types.is_float_dtype(X[col])]
        X = X.drop(columns=self.id_like_cols_, errors="ignore")

        # drop duplicates from [X|y]
        XY = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        XY = XY.drop_duplicates().reset_index(drop=True)
        y_name = y.name if y.name is not None else "target"
        y = XY[y_name]
        X = XY.drop(columns=[y_name])
        self.task_type_ = self.task_type

        # drop constant columns
        nunique = X.nunique(dropna=False)
        self.constant_cols_ = nunique[nunique <= 1].index.tolist()
        X = X.drop(columns=self.constant_cols_, errors="ignore")

        # drop ID-like columns (almost all unique)
        nunique = X.nunique(dropna=False)
        n_rows = len(X)
        new_candidates_id_like = nunique[(nunique > 1) & (nunique >= n_rows * self.id_like_threshold)].index.tolist()
        new_id_like = [col for col in new_candidates_id_like if not pd.api.types.is_float_dtype(X[col])]
        self.id_like_cols_ = list(set(self.id_like_cols_ + new_id_like))
        X = X.drop(columns=new_id_like, errors="ignore")

        # split by dtype after pruning
        self.raw_cols_ = X.columns.tolist()
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = [c for c in self.raw_cols_ if c not in self.num_cols_]

        if self.cat_cols_:
            X[self.cat_cols_] = X[self.cat_cols_].astype(str)

        # imputers
        self.num_imputer_ = SimpleImputer(strategy="mean")
        self.cat_imputer_ = SimpleImputer(strategy="most_frequent") if self.cat_cols_ else None

        X_num = pd.DataFrame(self.num_imputer_.fit_transform(X[self.num_cols_]) if self.num_cols_ else np.empty((len(X), 0)),
                             columns=self.num_cols_, index=X.index)
        if self.cat_cols_:
            X_cat = pd.DataFrame(self.cat_imputer_.fit_transform(X[self.cat_cols_]), columns=self.cat_cols_, index=X.index)
        else:
            X_cat = pd.DataFrame(index=X.index)

        # target encoder for categoricals
        if len(self.cat_cols_) > 0:
            if self.task_type_ == "classification":
                tgt_type = "binary" if y.nunique() <= 2 else "multiclass"
            else: 
                tgt_type = "continuous"

            self.target_encoder_ = TargetEncoder(target_type=tgt_type, random_state=self.random_state)
            X_cat_enc_np = self.target_encoder_.fit_transform(X_cat, y)
            
            self.encoded_cat_names_ = self.target_encoder_.get_feature_names_out(self.cat_cols_)
            X_cat_enc = pd.DataFrame(X_cat_enc_np, index=X.index, columns=self.encoded_cat_names_)
        else:
            X_cat_enc = pd.DataFrame(index=X.index)
            self.encoded_cat_names_=[]

        # combine numeric + encoded categoricals
        X_enc = pd.concat([X_num, X_cat_enc], axis=1)
        self.encoded_feature_names_ = X_enc.columns.tolist()

        # scale numeric features (and optionally the encoded categoricals too, since they’re numeric)
        self.scaler_ = StandardScaler()
        if self.scale_all_numeric_after_encoding:
            X_scaled = pd.DataFrame(self.scaler_.fit_transform(X_enc), index=X_enc.index, columns=X_enc.columns)
        else:
            # scale only original numeric columns; leave encoded categoricals as-is
            X_scaled = X_enc.copy()
            if self.num_cols_:
                X_scaled[self.num_cols_] = self.scaler_.fit_transform(X_enc[self.num_cols_])

        # encode target
        if self.task_type_ == "classification":
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(y)
        elif self.task_type_ == "regression" and self.scale_target_in_regression:
            self.y_scaler_ = StandardScaler()
            self.y_scaler_.fit(y.values.reshape(-1, 1))

        # report
        self.keep_cols_ = self.encoded_feature_names_
        self.report_ = PreprocessReport(
            dropped_constant=self.constant_cols_,
            dropped_id_like=self.id_like_cols_,
            kept_raw_columns=self.raw_cols_,
            numeric_columns=self.num_cols_,
            categorical_columns=self.cat_cols_,
            task_type=self.task_type_,
        )
        return self

    def transform(
            self,
            X: pd.DataFrame,
            y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Apply transforms to the data using the fitted preprocessing pipeline.

        Args:
            X (pd.DataFrame): Data to transform.
            y (Optional[pd.Series]): Target variables.

        Returns:
            Tuple[pd.DataFrame, Optional[np.ndarray]]:
                - The transformed data.
                - Transformed target variables.
        """
        self._check_is_fitted()
        X = X.copy()

        # keep only columns seen during fit; drop unseen columns; add missing as NaN
        for c in self.constant_cols_ + self.id_like_cols_:
            if c in X.columns:
                X = X.drop(columns=[c])

        # align columns to training raw columns
        for c in self.raw_cols_:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.raw_cols_]

        # drop duplicates 
        if y is not None:
            y = y.copy()
            y_name = y.name if y.name is not None else "target"
            XY = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            XY = XY.drop_duplicates(keep='first').reset_index(drop=True)
            y = XY[y_name]
            X = XY.drop(columns=[y_name])
        else:
            X = X.drop_duplicates(keep='first').reset_index(drop=True)

        # split
        X_num = X[self.num_cols_] if self.num_cols_ else pd.DataFrame(index=X.index)
        X_cat = X[self.cat_cols_] if self.cat_cols_ else pd.DataFrame(index=X.index)

        if self.cat_cols_:
            X_cat = X_cat.astype(str)

        # impute
        if self.num_cols_:
            X_num = pd.DataFrame(self.num_imputer_.transform(X_num), columns=self.num_cols_, index=X.index)
        if self.cat_cols_:
            X_cat = pd.DataFrame(self.cat_imputer_.transform(X_cat), columns=self.cat_cols_, index=X.index)

        # encode categoricals
        if self.cat_cols_:
            X_cat_enc = pd.DataFrame(self.target_encoder_.transform(X_cat), index=X.index, columns=self.encoded_cat_names_)
        else:
            X_cat_enc = pd.DataFrame(index=X.index)
        X_enc = pd.concat([X_num, X_cat_enc], axis=1)

        # align encoded feature names (in case columns order differs)
        X_enc = X_enc.reindex(columns=self.encoded_feature_names_, fill_value=0.0)

        # scale
        if self.scale_all_numeric_after_encoding:
            X_scaled = pd.DataFrame(self.scaler_.transform(X_enc), index=X_enc.index, columns=X_enc.columns)
        else:
            X_scaled = X_enc.copy()
            if self.num_cols_:
                X_scaled[self.num_cols_] = self.scaler_.transform(X_enc[self.num_cols_])

        # target handling
        y_out = None
        if y is not None:
            y = y.copy()
            if self.task_type_ == "classification":
                y_out = self.label_encoder_.transform(y)
            elif self.task_type_ == "regression" and self.y_scaler_ is not None:
                y_out = self.y_scaler_.transform(y.values.reshape(-1, 1)).ravel()
            else:
                y_out = y.values

        return X_scaled, y_out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Performs fit and transform.
        """
        return self.fit(X, y).transform(X, y)

    def _validate_inputs(self, X, y):
        """
        Validates input X and y.

        Args:
            X: Input dataset (DataFrame or array-like).
            y: Target variables (Series or array-like).

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: (X, y)
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.Series(y, name="target")
        if isinstance(y, pd.DataFrame):
            assert y.shape[1] == 1, "y must be 1D"
            y = y.iloc[:, 0]
        if y.name is None:
            y.name = "target"

        X = X.copy()
        y = y.copy()

        # convert object columns to numeric if possible
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')

        # check target validity for regression
        if self.task_type == 'regression':
            y = pd.to_numeric(y, errors='coerce')
            y = y.astype(np.float64) 
            valid_indices = y.notna()
            if not valid_indices.all():
                X = X.loc[valid_indices].reset_index(drop=True)
                y = y.loc[valid_indices].reset_index(drop=True)

        return X, y

    def _check_is_fitted(self):
        if self.report_ is None or self.keep_cols_ is None:
            raise RuntimeError("Preprocessor is not fitted yet. Call fit() first.")

    def get_report(self) -> PreprocessReport:
        """
        Generate report after the fitting process.
        """
        self._check_is_fitted()
        return self.report_
