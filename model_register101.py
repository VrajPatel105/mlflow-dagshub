from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import mlflow
import mlflow.sklearn
# import dagshub
# dagshub.init(repo_owner='VrajPatel105', repo_name='mlflow-dagshub', mlflow=True)


# mlflow.set_tracking_uri("https://dagshub.com/VrajPatel105/mlflow-dagshub.mlflow")

df = pd.read_csv(
    "https://raw.githubusercontent.com/npradaschnor/"
    "Pima-Indians-Diabetes-Dataset/refs/heads/master/diabetes.csv"
)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

mlflow.set_experiment("understand model registry")

with mlflow.start_run():

    grid_search.fit(X_train, y_train)

    # logging all the childern by having a inner with statement
    for i in range(len(grid_search.cv_results_['params'])):

        with mlflow.start_run(nested=True) as child:

            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric(
                                "mean_test_score",
                                grid_search.cv_results_["mean_test_score"][i]
                            )

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", best_score)

    train_df = X_train.copy()
    train_df["Outcome"] = y_train

    test_df = X_test.copy()
    test_df["Outcome"] = y_test

    train_data = mlflow.data.from_pandas(train_df)
    test_data = mlflow.data.from_pandas(test_df)

    mlflow.log_input(train_data, "training")
    mlflow.log_input(test_data, "validation")
    signature = mlflow.models.infer_signature(X_train, grid_search.best_estimator_.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=grid_search.best_estimator_,
        artifact_path="random_forest", 
        signature=signature
    )

    mlflow.set_tag("author", "Vraj Patel")

    print(best_params)
    print(best_score)