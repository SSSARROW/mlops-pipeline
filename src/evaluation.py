import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,f1_score

class EvaluationStrategy(ABC):
    """Abstract base class for evaluation strategies."""

    @abstractmethod
    def evaluate(self, y_true:np.ndarray, y_pred:np.ndarray):
        """Evaluate the model on the test data.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        pass

class MSE(EvaluationStrategy):
    """Concrete strategy for Mean Squared Error evaluation."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) :
        """Calculate Mean Squared Error."""
        try:
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error calculating Mean Squared Error: {e}")
            raise e

class R2Score(EvaluationStrategy):
    """Concrete strategy for R2 Score evaluation."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) :
        """Calculate R2 Score."""
        try:
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error calculating R2 Score: {e}")
            raise e

class F1Score(EvaluationStrategy):
    """Concrete strategy for F1 Score evaluation."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) :
        """Calculate F1 Score."""
        try:
            f1 = f1_score(y_true, y_pred, average='weighted')
            logging.info(f"F1 Score: {f1}")
            return f1
        except Exception as e:
            logging.error(f"Error calculating F1 Score: {e}")
            raise e
