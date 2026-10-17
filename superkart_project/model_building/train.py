
# -----------------------------
# Data manipulation
# -----------------------------
import pandas as pd

# -----------------------------
# Preprocessing
# -----------------------------
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# -----------------------------
# Model training and tuning
# -----------------------------
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# -----------------------------
# Model evaluation
# -----------------------------
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# -----------------------------
# Model serialization
# -----------------------------
import joblib

# -----------------------------
# MLflow
# -----------------------------
import mlflow


# ============================================================
# MLflow configuration
# ============================================================

# MLflow UI is running locally on port 5000
mlflow.set_tracking_uri("http://localhost:5000")

# Create/select the MLflow experiment
mlflow.set_experiment("superkart-training-experiment")


# ============================================================
# Load data prepared by prep.py
# ============================================================

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")

ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)
print("ytrain shape:", ytrain.shape)
print("ytest shape:", ytest.shape)


# ============================================================
# Identify feature types
# ============================================================

numeric_features = Xtrain.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = Xtrain.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# Preprocessing
# ============================================================

preprocessor = make_column_transformer(
    (
        StandardScaler(),
        numeric_features
    ),
    (
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
)


# ============================================================
# Base XGBoost regression model
# ============================================================

xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)


# ============================================================
# Hyperparameter grid
# ============================================================

param_grid = {
    "xgbregressor__n_estimators": [50, 100],
    "xgbregressor__max_depth": [2, 3],
    "xgbregressor__learning_rate": [0.05, 0.1],
}


# ============================================================
# Create complete ML pipeline
# ============================================================

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)


# ============================================================
# Start MLflow run
# ============================================================

with mlflow.start_run():

    print("\nStarting hyperparameter tuning...")

    # GridSearchCV for regression
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain)

    print("\nHyperparameter tuning complete.")

    # ========================================================
    # Log all hyperparameter combinations
    # ========================================================

    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        param_set = results["params"][i]

        # GridSearchCV returns negative RMSE
        mean_rmse = -results["mean_test_score"][i]

        std_rmse = results["std_test_score"][i]

        with mlflow.start_run(nested=True):

            mlflow.log_params(param_set)

            mlflow.log_metric(
                "mean_cv_rmse",
                mean_rmse
            )

            mlflow.log_metric(
                "std_cv_rmse",
                std_rmse
            )

    # ========================================================
    # Best parameters
    # ========================================================

    print("\nBest parameters:")
    print(grid_search.best_params_)

    mlflow.log_params(
        grid_search.best_params_
    )

    # ========================================================
    # Best model
    # ========================================================

    best_model = grid_search.best_estimator_


    # ========================================================
    # Predictions
    # ========================================================

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)


    # ========================================================
    # Evaluation metrics
    # ========================================================

    train_mae = mean_absolute_error(
        ytrain,
        y_pred_train
    )

    test_mae = mean_absolute_error(
        ytest,
        y_pred_test
    )

    train_rmse = mean_squared_error(
        ytrain,
        y_pred_train
    ) ** 0.5

    test_rmse = mean_squared_error(
        ytest,
        y_pred_test
    ) ** 0.5

    train_r2 = r2_score(
        ytrain,
        y_pred_train
    )

    test_r2 = r2_score(
        ytest,
        y_pred_test
    )


    # ========================================================
    # Print results
    # ========================================================

    print("\n==============================")
    print("Model Evaluation")
    print("==============================")

    print("\nTraining metrics:")
    print("MAE :", train_mae)
    print("RMSE:", train_rmse)
    print("R2  :", train_r2)

    print("\nTesting metrics:")
    print("MAE :", test_mae)
    print("RMSE:", test_rmse)
    print("R2  :", test_r2)


    # ========================================================
    # Log metrics to MLflow
    # ========================================================

    mlflow.log_metrics({

        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2": train_r2,

        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2

    })


    # ========================================================
    # Save model
    # ========================================================

    model_path = (
        "superkart_project/deployment/"
        "best_superkart_sales_model_v1.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print("\nModel saved to:")
    print(model_path)


    # ========================================================
    # Log model as MLflow artifact
    # ========================================================

    mlflow.log_artifact(
        model_path,
        artifact_path="model"
    )

    print("\nModel successfully logged to MLflow.")


print("\nTraining completed successfully!")
