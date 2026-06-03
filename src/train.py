import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from model_training import (
    prepare_model_data,
    split_data,
    train_logistic_regression,
    train_random_forest,
    evaluate_model,
)


def train_and_track_models(
    data_path="data/processed/rfm_labeled.csv"
):
    # Load data
    df = pd.read_csv(data_path)

    # Prepare features and target
    X, y = prepare_model_data(df)

    # Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = {
        "Logistic Regression": train_logistic_regression(
            X_train,
            y_train
        ),
        "Random Forest": train_random_forest(
            X_train,
            y_train
        ),
    }

    results = []

    best_model = None
    best_model_name = None
    best_auc = 0

    mlflow.set_experiment(
        "credit-risk-model"
    )

    for model_name, model in models.items():

        with mlflow.start_run(
            run_name=model_name
        ):

            metrics = evaluate_model(
                model,
                X_test,
                y_test
            )

            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                model,
                artifact_path=model_name
            )

            results.append({
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"]
            })

            if metrics["f1_score"] > best_auc:
                best_auc = metrics["f1_score"]
                best_model = model
                best_model_name = model_name

    results_df = pd.DataFrame(results)

    os.makedirs(
        "report",
        exist_ok=True
    )

    results_df.to_csv(
        "report/final_model_results.csv",
        index=False
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )

    print("Training complete.")
    print(
        f"Best model: {best_model_name}"
    )

    print(results_df)

    return (
        results_df,
        best_model_name
    )


if __name__ == "__main__":
    train_and_track_models(
        data_path="data/processed/rfm_labeled.csv"
    )