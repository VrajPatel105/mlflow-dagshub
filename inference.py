# inference

import mlflow.pyfunc
import pandas as pd


data = pd.DataFrame(
    [[1, 85, 66, 29, 0, 26.6, 0.351, 31]],
    columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ],
)


model_name = "diabetes-rf"
model_version = 1


model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)


prediction = model.predict(data)

print(prediction)


# output 
# (mlenv) vraj@Vraj:/mnt/c/dev/temp/mlops-mlflow$ python inference.py
# [0]