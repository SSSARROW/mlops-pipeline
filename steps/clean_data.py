import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataProcessor
from typing_extensions import Annotated
from typing import Tuple
from scipy.sparse import csr_matrix
import numpy as np

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[csr_matrix, "X_train"],
    Annotated[csr_matrix, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
]:
    """
    Cleans the data and divides it into train and test sets.

    Args:
        df: Raw data.
    Returns:
        A tuple of processed and split data.
    """
    try:
        processor = DataProcessor()
        X_train, X_test, y_train, y_test = processor.process(df)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"An error occurred during data cleaning: {e}")
        raise e
    