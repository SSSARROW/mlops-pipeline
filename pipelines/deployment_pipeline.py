import json
import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, SAGEMAKER
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
# from zenml.steps import  Output
from pydantic import BaseModel, _dynamic_imports
import mlflow
import mlflow.sklearn  # or the appropriate flavor

from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_data
from steps.config import ModelNameConfig
from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data

class DeploymentTriggerConfig(BaseModel):
    """
    Configuration for the deployment trigger.
    """
    min_accuracy: float
@step
def deployment_trigger(
    accuracy: float,
    config:DeploymentTriggerConfig
):
    """
    Trigger for deployment based on model accuracy.
    """
    if accuracy >= config.min_accuracy:
        return True
    return False
class MLFlowDeploymentLoaderStepParameters(BaseModel):
    pipeline_name: str
    step_name: str
    running : bool = True

@step
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """
    Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: Name of the pipeline that started the service.
        pipeline_step_name: Name of the step that started the service.
        running: Whether to return a running service or not.
        model_name: Name of the model to be served.
    """
    #get the Mlflow stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()

    #fetch existing services with same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )
    if not existing_services:
        raise RuntimeError(
            f"No running service found for pipeline '{pipeline_name}', "
            f"step '{pipeline_step_name}', and model '{model_name}'."
            f"pipeline for the '{model_name} model is currently running"
            f"running."
        )
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray,
    config: DeploymentTriggerConfig,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns", None)
    data.pop("index", None)

    # Replace these with the actual columns from your train.csv
    columns_for_df = [
        "Date",
        "Location",
        "MinTemp",
        "MaxTemp",
        "Rainfall",
        "Evaporation",
        "Sunshine",
        "WindGustDir",
        "WindGustSpeed",
        "WindDir9am",
        "WindDir3pm",
        "WindSpeed9am",
        "WindSpeed3pm",
        "Humidity9am",
        "Humidity3pm",
        "Pressure9am",
        "Pressure3pm",
        "Cloud9am",
        "Cloud3pm",
        "Temp9am",
        "Temp3pm",
        "RainToday",
        # Add or remove columns as per your model's requirements
    ]

    df = pd.DataFrame(data["data"], columns=columns_for_df)
    # If your model expects only numeric columns, select them here
    # df = df.select_dtypes(include=[np.number])

    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction
    
@pipeline(
    enable_cache=False,
    settings={"docker": docker_settings}  # <-- wrap in dict with key "docker"
)
def continous_deployment_pipeline(
    min_accuracy: float = 0.04,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_data(data_path="data/train.csv")
    X_train, X_test, y_train, y_test = clean_data(df)
    
    config = ModelNameConfig()
    model = train_model(X_train=X_train, y_train=y_train, config=config)
    
    f1_score = evaluate_model(model=model, X_test=X_test, y_test=y_test,)
    # Pass DeploymentTriggerConfig to deployment_trigger
    deploy_decision = deployment_trigger(
        accuracy=f1_score,  # <-- CORRECT
        config=DeploymentTriggerConfig(min_accuracy=min_accuracy)
    )
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deploy_decision,
        workers=workers,
        timeout=timeout,
    )


@pipeline(enable_cache=True, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str, min_accuracy: float = 0.0):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=True,
    )
    predictions = predictor(
        service=service,
        data=data,
        config=DeploymentTriggerConfig(min_accuracy=min_accuracy)  # Pass an instance, not the class
    )
    return predictions

# Set experiment
mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    # Train your m
    # model
    model = train_model(X_train, y_train)
    
    # Log the model to MLflow
    mlflow.sklearn.log_model(model, "model")
