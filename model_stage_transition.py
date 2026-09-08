from mlflow.tracking import MlflowClient

# Initialize the MLflow Client
client = MlflowClient()

# Define the model name and version
model_name = "diabetes-rf"
model_version = 1 

# Transition the model version to a new stage
new_stage = "Staging"  # Possible values: "None", "Staging", "Production", "Archived"
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=True
)

print(f"Model version {model_version} transitioned to {new_stage}")