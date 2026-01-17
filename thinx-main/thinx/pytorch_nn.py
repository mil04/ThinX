import numpy as np
import pandas as pd
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Union
from sklearn.base import BaseEstimator
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class BasicNeuralNetwork(nn.Module):
    """
    Simple feedforward neural network with two hidden layers.
    Architecture: Input -> Linear(100) -> ReLU -> Linear(100) -> ReLU -> Linear(Output)

    Args:
        n_inputs (int): Number of input features.
        n_outputs (int): Number of output features.

    Methods:
        forward(x): Performs a forward pass.
    """
    def __init__(self, n_inputs: int, n_outputs: int):
        if n_inputs < 0 or n_outputs < 0:
            raise ValueError("n_inputs and n_outputs must be positive")

        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_outputs)
        )
    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.network(x)
        
        
class PyTorchNN(BaseEstimator):
    """
    PyTorch Neural Network compatible with scikit-learn.

    Architecture: Input -> Linear(100) -> ReLU -> Linear(100) -> ReLU -> Linear(Output)

    Args:
        task_type (str): "classification" or "regression".
        epochs (int): Maximum number of training epochs.
        lr (float): Learning rate for Adam optimizer.
        batch_size (int): Batch size.
        random_state (int): Seed for reproducibility.

    Attributes:
        model_ (Optional[nn.Module]): Trained PyTorch model after calling `fit`.
    """
    def __init__(
        self,
        task_type: str = 'classification',
        epochs: int = 1000,
        lr: float = 0.001,
        batch_size: int = 512,
        random_state: int = 42
    ):
        self.task_type = task_type
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.model_ = None
        torch.manual_seed(self.random_state)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        verbose: bool = False
    ) -> 'PyTorchNN':
        """
        Train the neural network with early stopping.

        Args:
            X (pd.DataFrame or np.ndarray): Training data.
            y (np.ndarray): Training target values.
            verbose (bool): Indicator of printing progress during training.

        Returns:
            PyTorchNN: Fitted estimator.

        Raises:
            ValueError:
                - If X or y is None or empty.
                - If X and y have different number of samples.
                - If task_type is not 'classification' or 'regression'.
        """
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            raise ValueError("X and y cannot be None or empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        n_inputs = X_train.shape[1]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

        if self.task_type == 'classification':
            n_outputs = len(np.unique(y)) 
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()
        elif self.task_type == 'regression':
            n_outputs = 1
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
            criterion = nn.MSELoss()
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")

        # --- Define the model and Adam optimizer ---
        self.model_ = BasicNeuralNetwork(n_inputs, n_outputs)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        # --- Prepare the data ---
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        patience = 10  # How many epochs to wait for improvement before stopping
        patience_counter = 0
        loss = None
        best_val_loss = float('inf')
        best_model_state = None

        # --- Perform training with early stopping ---
        for epoch in range(self.epochs):
            self.model_.train() 
            for batch_X, batch_y in train_loader:
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # --- Validation Step ---
            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

            if verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                print(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

            # --- Early Stopping Logic ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model_.state_dict())
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # --- Get the best model ---
        if best_model_state:
            self.model_.load_state_dict(best_model_state)
        
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get predictions for X using the trained model.

        Args:
            X (pd.DataFrame or np.ndarray): Data to make predictions for.

        Returns:
            np.ndarray: Predicted labels (classification) or values (regression).

        Raises:
            RuntimeError: If called before the model is trained (fit not called).
            ValueError: If X is None or empty.
        """
        if self.model_ is None:
            raise RuntimeError("You must call fit before calling predict.")

        if X is None or len(X) == 0:
            raise ValueError("X cannot be None or empty")
            
        X_tensor = torch.tensor(X, dtype=torch.float32)

        self.model_.eval() 
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            if self.task_type == 'classification':
                _, predicted = torch.max(outputs.data, 1)
                return predicted.numpy()
            else: # regression case
                return outputs.numpy().flatten()

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute class probabilities for classification tasks.

        Args:
            X (pd.DataFrame or np.ndarray): Input features.

        Returns:
            np.ndarray: Class probabilities, shape (n_samples, n_classes).

        Raises:
            AttributeError: If called for a regression task.
            RuntimeError: If called before the model is trained (fit not called).
            ValueError: If X is None or empty.
        """
        if self.task_type != 'classification':
            raise AttributeError(f"predict_proba is not available for task_type='{self.task_type}'")

        if self.model_ is None:
            raise RuntimeError("You must call fit before calling predict_proba.")

        if X is None or len(X) == 0:
            raise ValueError("X cannot be None or empty")

        X_tensor = torch.tensor(X, dtype=torch.float32)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probabilities = torch.softmax(logits, dim=1)
            return probabilities.numpy()