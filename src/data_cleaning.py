import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from typing import Tuple
from scipy.sparse import csr_matrix

class DataProcessor:
    """
    Processes the data by cleaning, splitting, preprocessing, and resampling it.
    """

    def process(self, data: pd.DataFrame) -> Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
        """
        Executes the full data processing pipeline.
        
        Args:
            data: The input pandas DataFrame.
            
        Returns:
            A tuple containing resampled training features, test features,
            resampled training labels, and test labels.
        """
        try:
            # Drop target variable and separate features (X) and target (y)
            X = data.drop(columns=["RainTomorrow"])
            y = data["RainTomorrow"]
            
            # Feature engineering for dates
            if "Date" in X.columns:
                date_parts = pd.to_datetime(X["Date"])
                X["Year"] = date_parts.dt.year
                X["Month"] = date_parts.dt.month
                X["Day"] = date_parts.dt.day
                X = X.drop(columns=["Date"])
            
            # Initial train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Define categorical and numerical features
            categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
            numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()

            # Create preprocessing pipelines for numerical and categorical features
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ])
            
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
            ])
            
            # Create the main preprocessor with ColumnTransformer
            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numerical_features),
                ("cat", cat_pipeline, categorical_features)
            ], remainder='passthrough')
            
            # Fit on training data and transform both train and test data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Scale the features
            scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse matrices
            X_train_scaled = scaler.fit_transform(X_train_processed)
            X_test_scaled = scaler.transform(X_test_processed)
            
            # Apply SMOTE to the training data to handle class imbalance
            logging.info("Applying SMOTE to the training data.")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            
            logging.info("Data processing complete.")
            return X_train_resampled, X_test_scaled, y_train_resampled.to_numpy(), y_test.to_numpy()
            
        except Exception as e:
            logging.error(f"Error in data processing: {e}")
            raise e



                    
                