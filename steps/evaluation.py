import logging
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from zenml import step
from src.evaluation import MSE, R2Score, F1Score
from src.model_dev import RainNet
import mlflow

@step(experiment_tracker="mlflow_tracker")
def evaluate_model(model: RainNet, X_test: csr_matrix, y_test: np.ndarray) -> float:
    """Evaluate the trained model on test data.
    
    Args:
        model: Trained model to evaluate.
        X_test: Test features.
        y_test: Test labels.
    Returns:
        F1 score as a float.
    """
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Convert test data to tensor
        X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
        
        # Make predictions
        with torch.no_grad():
            predictions = model(X_test_tensor)
            # Apply sigmoid since the model output is logits
            predictions_proba = torch.sigmoid(predictions).cpu().numpy().flatten()
        
        # Convert probabilities to binary classes for F1 score
        y_pred_binary = (predictions_proba > 0.5).astype(int)
        
        # Evaluate using different metrics
        mse_strategy = MSE()
        mse_score = mse_strategy.evaluate(y_test, predictions_proba)
        mlflow.log_metric("mse", mse_score)
        
        r2_strategy = R2Score()
        r2_score = r2_strategy.evaluate(y_test, predictions_proba)
        mlflow.log_metric("r2_score", r2_score)
        
        f1_strategy = F1Score()
        f1_score = f1_strategy.evaluate(y_test, y_pred_binary)
        mlflow.log_metric("f1_score", f1_score)
        
        logging.info(f"Model Evaluation Results:")
        logging.info(f"  MSE: {mse_score:.4f}")
        logging.info(f"  R2 Score: {r2_score:.4f}")
        logging.info(f"  F1 Score: {f1_score:.4f}")
        
        return float(f1_score)
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise e