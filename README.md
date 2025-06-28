# Rain Prediction MLOps Pipeline

This project implements a complete MLOps pipeline for rain prediction using ZenML, PyTorch, and scikit-learn.

## Project Structure

```
├── src/
│   ├── model_dev.py          # PyTorch neural network model (RainNet)
│   ├── data_cleaning.py      # Data preprocessing strategies
│   └── evaluation.py         # Model evaluation metrics
├── steps/
│   ├── ingest_data.py        # Data ingestion step
│   ├── clean_data.py         # Data cleaning and splitting step
│   ├── model_train.py        # Model training step
│   ├── evaluation.py         # Model evaluation step
│   └── config.py             # Model configuration
├── pipelines/
│   └── training_pipeline.py  # Main training pipeline
├── data/
│   └── train.csv             # Training dataset
├── run_pipeline.py           # Pipeline runner
└── requirements.txt          # Project dependencies
```

## Features

- **RainNet**: Custom PyTorch neural network with batch normalization and dropout
- **Data Preprocessing**: Automated data cleaning, missing value handling, and categorical encoding
- **MLOps Pipeline**: Complete ZenML pipeline with data ingestion, preprocessing, training, and evaluation
- **Model Evaluation**: Multiple metrics (MSE, R², F1-Score) for comprehensive model assessment
- **Strategy Pattern**: Clean separation of concerns using abstract classes

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize ZenML**:
   ```bash
   zenml init
   ```

3. **Run the Pipeline**:
   ```bash
   python run_pipeline.py
   ```

## Model Architecture

The RainNet model consists of:
- Input layer with dynamic size based on features
- 3 hidden layers (128 → 64 → 64 → 1 neurons)
- Batch normalization for training stability
- ReLU activation functions
- Dropout layers for regularization
- Binary classification output with sigmoid activation

## Pipeline Flow

1. **Data Ingestion**: Load CSV data from specified path
2. **Data Cleaning**: 
   - Remove target and date columns
   - Handle missing values (median for numerical, mode for categorical)
   - Encode categorical variables
3. **Data Splitting**: Split into train/test sets (80/20)
4. **Model Training**: Train RainNet with PyTorch
5. **Model Evaluation**: Calculate MSE, R², and F1 scores

## Configuration

Model configuration can be modified in `steps/config.py`:
- Model name selection
- Training parameters (epochs, batch size)
- Learning rate and optimizer settings

## Monitoring

The pipeline includes comprehensive logging for:
- Data processing steps
- Training progress and loss
- Model evaluation metrics
- Error handling and debugging

## Future Enhancements

- Model versioning and artifact storage
- Hyperparameter tuning
- Model deployment pipeline
- Real-time prediction API
- A/B testing framework 