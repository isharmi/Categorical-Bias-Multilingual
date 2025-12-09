# nested_optuna_cv.py

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from tqdm import tqdm

from models.predictive_model import BiasPredictor
from evaluation.metrics import Evaluator


############################################################
#  STEP 1: TUNE HYPERPARAMETERS ONCE ON FULL TRAINING SET  #
############################################################
def tune_hyperparameters_once(X, y, model_name, n_trials=25, random_state=42):
    """
    Tune hyperparameters ONCE using Optuna with 5-fold CV.
    """

    def objective(trial):
        # Define hyperparameter search space
        if model_name == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "random_state": random_state
            }

        elif model_name == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "random_state": random_state
            }

        elif model_name == "lgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "random_state": random_state
            }

        elif model_name == "cat":
            params = {
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "iterations": trial.suggest_int("iterations", 100, 600),
                "random_state": random_state,
                "verbose": False
            }

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # 5-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        mse_list = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = BiasPredictor(model_name=model_name, **params)
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            mse = np.mean((preds - y_val) ** 2)
            mse_list.append(mse)

        return np.mean(mse_list)

    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"\nBest params for {model_name}:")
    print(study.best_params)
    print(f"Best CV MSE: {study.best_value:.4f}")

    return study.best_params, study.best_value
    


##########################################################
#  STEP 2: LOPOCV USING THE FROZEN BEST HYPERPARAMETERS  #
##########################################################
def lopo_cv_with_frozen_params(X, y, model_name, best_params):
    """
    Perform Leave-One-Put-Out CV using the best tuned hyperparameters.
    """

    n_samples = X.shape[0]
    preds = np.zeros(n_samples)

    for test_idx in tqdm(range(n_samples), desc=f"LOPOCV for {model_name}"):
        # Split
        X_train = np.delete(X, test_idx, axis=0)
        y_train = np.delete(y, test_idx, axis=0)
        X_test = X[test_idx].reshape(1, -1)

        # Train with frozen params
        model = BiasPredictor(model_name=model_name, **best_params)
        model.fit(X_train, y_train)

        preds[test_idx] = model.predict(X_test)

    # Evaluate
    metrics = Evaluator.regression_metrics(y, preds)
    print("\nLOPOCV Results:")
    print(metrics)

    return metrics, preds



    import shap
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if the model is tree-based
    tree_models = ["rf", "xgb", "lgbm", "cat"]
    if hasattr(model, "model_name") and model.model_name in tree_models:
        # TreeExplainer works natively with tree models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        # Wrap predict for non-tree models as a callable returning array
        def predict_fn(X_input):
            X_input = np.atleast_2d(X_input)  # ensure 2D
            return np.array([model.predict(x) for x in X_input])

        background = X[:50]  # small background for KernelExplainer
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X)

    # Compute mean absolute SHAP values per feature
    shap_dict = {f: float(np.mean(np.abs(shap_values[:, i]))) for i, f in enumerate(feature_names)}

    # Save summary plot
    plot_path = os.path.join(output_dir, f"shap_summary_{getattr(model, 'model_name', 'model')}.png")
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.savefig(plot_path)
    plt.close()

    return shap_dict, plot_path
