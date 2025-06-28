import logging
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix
import numpy as np
from zenml import step
from .config import ModelNameConfig
from src.model_dev import RainNet

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def train_model(
    X_train: csr_matrix,
    y_train: np.ndarray,
    config: ModelNameConfig,
    epochs: int = 50,
    batch_size: int = 64
) -> RainNet:
    """ 
    Trains the model with ingested data and advanced techniques.
    Args:
        X_train: Training data (features).
        y_train: Training labels.
        config: Model configuration.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
    Returns:
        Trained model.
    """
    try:
        # Create a validation set from the training data
        X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Convert data to tensors
        X_train_tensor = torch.tensor(X_train_part.toarray(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_part, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val_part.toarray(), dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_part, dtype=torch.float32).unsqueeze(1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # Initialize model
        if config.model_name == "RainNet":
            model = RainNet(X_train.shape[1])
        else:
            raise ValueError(f"Model {config.model_name} is not supported.")
        
        # Setup training (optimizer, scheduler, criterion)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        mlflow.pytorch.autolog()
        logging.info("Starting model training...")
        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_tensor)
                val_preds_cls = (torch.sigmoid(val_preds) > 0.5).int()
                f1 = f1_score(y_val_tensor.cpu(), val_preds_cls.cpu())
            
            scheduler.step(f1)
            
            if (epoch + 1) % 5 == 0:
                logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Val F1: {f1:.4f}')
        
        logging.info("Model training completed successfully.")
        return model
        
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e