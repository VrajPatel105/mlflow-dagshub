from mlflow.tracking import MlflowClient
import mlflow

# Initialize the MLflow Client
client = MlflowClient()


model_path = "mlruns/4/models/m-d93eb4b3214d4895b1850a1a1417b090/artifacts/MLmodel"


# Construct the model URI
run_id = "1c975acf64824ea197a84717eedadef7"

# Use the MLflow logged model ID, not the local mlruns path
model_uri = "models:/m-d93eb4b3214d4895b1850a1a1417b090"


# Register the model in the model registry
model_name = "diabetes-rf"
result = mlflow.register_model(model_uri, model_name)


import time
time.sleep(5)


# Add a description to the registered model version
client.update_model_version(
    name=model_name,
    version=result.version,
    description="This is a RandomForest model trained to predict diabetes outcomes based on Pima Indians Diabetes Dataset."
)


client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="experiment by Vraj",
    value="diabetes prediction for model registry ::))))"
)


print(f"Model registered with name: {model_name} and version: {result.version}")
print(f"Added tags to model {model_name} version {result.version}")


# Get and print the registered model information
registered_model = client.get_registered_model(model_name)
print("Registered Model Information:")
print(f"Name: {registered_model.name}")
print(f"Creation Timestamp: {registered_model.creation_timestamp}")
print(f"Last Updated Timestamp: {registered_model.last_updated_timestamp}")
print(f"Description: {registered_model.description}")