import os
import json
import pickle
import numpy as np
import pandas as pd
from data.wals_loader import WALSLoader
from models.predictive_model import BiasPredictor
from hyperparameter_tuning.tuner import tune_hyperparameters_once, lopo_cv_with_frozen_params

from tqdm import tqdm


# ----------------------------
# Output directory
# ----------------------------
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Step 0: Compute bias scores
# ----------------------------
def compute_bias_scores_pipeline():
    loader = WALSLoader()

    loader.values = pd.read_csv("./data/values.csv")
    loader.codes = pd.read_csv("./data/codes.csv")
    loader.languages = pd.read_csv("./data/languages.csv")
    loader.parameters = pd.read_csv("./data/parameters.csv")
    print("WALS CSVs loaded successfully!")

    selected_langs = loader.languages["ID"].dropna().tolist()
    selected_features = loader.parameters["ID"].dropna().tolist()
    feature_cols = [c for c in loader.get_features_for_languages(selected_langs, selected_features).columns if c != "ID"]

    X = loader.get_features_for_languages(selected_langs, selected_features)
    X_clean = X[feature_cols].replace(-1, np.nan).fillna(0)
    X_numeric = X_clean.astype(float)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    # Bias score = distance to mean vector
    mean_vec = X_scaled.mean(axis=0)
    bias_scores = np.linalg.norm(X_scaled - mean_vec, axis=1)

    return X_scaled, bias_scores, selected_langs, feature_cols


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    print("Step 0: Computing bias scores...")
    X, y, languages, feature_names = compute_bias_scores_pipeline()
    
    models = ["rf", "xgb", "lgbm", "cat"]
    all_results = {}

    for model_name in models:
        print(f"\n=== Processing {model_name} ===")

        # Step 1: Tune hyperparameters ONCE
        print(f"\nStep 1: Tuning hyperparameters for {model_name}...")
        best_params, best_cv_mse = tune_hyperparameters_once(X, y, model_name=model_name, n_trials=30)

        # Step 2: Train final model on full data for SHAP
        final_model = BiasPredictor(model_name=model_name, **best_params)
        final_model.fit(X, y)

        # Step 3: LOPOCV with frozen parameters
        print(f"\nStep 2: LOPOCV with frozen parameters for {model_name}...")
        n_samples = X.shape[0]
        preds_per_sample = []

        for test_idx in tqdm(range(n_samples), desc=f"LOPOCV for {model_name}"):
            X_train = np.delete(X, test_idx, axis=0)
            y_train = np.delete(y, test_idx, axis=0)
            X_test = X[test_idx].reshape(1, -1)

            model = BiasPredictor(model_name=model_name, **best_params)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            preds_per_sample.append(pred[0])

        preds_per_sample = np.array(preds_per_sample)
        errors = preds_per_sample - y

        mse_per_sample = errors**2
        mae_per_sample = np.abs(errors)
        r2_per_sample = 1 - (mse_per_sample / np.var(y))

        metrics_summary = {
            "MSE": f"{mse_per_sample.mean():.4f} ± {mse_per_sample.std():.4f}",
            "MAE": f"{mae_per_sample.mean():.4f} ± {mae_per_sample.std():.4f}",
            "R2": f"{r2_per_sample.mean():.4f} ± {r2_per_sample.std():.4f}"
        }


        # Step 5: Save all results in dict
        all_results[model_name] = {
            "languages": languages,
            "bias_scores": y.tolist(),
            "predictions": preds_per_sample.tolist(),
            "metrics": metrics_summary,
            "best_params": best_params,
            "best_cv_mse": best_cv_mse
        }

        # Save model individually
        model_path = f"final_{model_name}_model.pkl"
        pickle.dump(final_model, open(model_path, "wb"))
        print(f"Saved final trained model for {model_name} to {model_path}")

    # Save all models results to a single JSON in current directory
    json_path = "full_results_all_models.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved all models' results to {json_path}")
    print("\nAll done!")


if __name__ == "__main__":
    main()
