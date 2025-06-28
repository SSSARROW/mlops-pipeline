import logging
import pandas as pd
from src.data_cleaning import DataProcessor

def get_data_for_test():
    try:
        df=pd.read_csv("data/train.csv")
        df = df.sample(n=100)
        data_cleaning = DataProcessor()
        df=data_cleaning.handle_data()
        df.drop(["review_score"],axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(f"Error in get_data_for_test: {e}")
        raise e