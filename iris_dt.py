import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Decision Tree classifier
max_depth = 1

mlflow.set_experiment("iris_dt")

# Apply mlflow to train
with mlflow.start_run():
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Log metric and parameter
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)

    # Create and save confusion-matrix plot
    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )

    plt.title("Decision Tree Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Log saved image file as an MLflow artifact
    mlflow.log_artifact(plot_path, artifact_path="plots")

    # Log the trained model
    mlflow.sklearn.log_model(
        sk_model=dt,
        name="decision_tree_model"
    )

    print(f"Accuracy: {accuracy:.4f}")

    # log code
    mlflow.log_artifact(__file__)

    # log model
    mlflow.sklearn.log_model(dt, "descision-tree-model")

    # set tag
    mlflow.set_tag('author', 'vraj')
    mlflow.set_tag('model', 'descision_tree')