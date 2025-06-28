import logging

import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting the data from the data_path.
    """
    def __init__(self,data_path: str):
        self.data_path = data_path

        """
        Args:
            data_path: path to the data file."""

    def get_data(self):
        """
        Ingesting the data from the data_path.
        """
        logging.info(f'Ingesting data from {self.data_path}')
        return pd.read_csv(self.data_path)

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """"
    Ingesting the data from the data_path.

    Args:
        data_path (str): The path to the data file.
    Returns:
        pd.DataFrame: The ingested data as a pandas DataFrame.
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error ingesting data: {e}")
        raise e

