# Categorical Bias in Multilingual NLP Models

A research project investigating categorical bias across multilingual NLP models using linguistic typology features from the **World Atlas of Language Structures (WALS)**. Multiple machine learning models are trained, tuned, and evaluated to predict and analyze bias scores across languages.

---

## 📁 Project Structure

```
Categorical-Bias-Multilingual/
├── data/                        # Raw and processed datasets
├── cldf-datasets-wals-0f5cd82/ # WALS linguistic typology dataset
├── models/                      # Saved model artifacts
├── evaluation/                  # Evaluation scripts and outputs
├── hyperparameter_tuning/       # Optuna tuning logs and configs
├── results/                     # Experiment results and outputs
├── catboost_info/               # CatBoost training logs
├── main.py                      # Main training and evaluation pipeline
├── rebuild_bias_scores.py       # Script to recompute bias scores
├── bias_scores.csv              # Computed bias scores per language
├── full_results_all_models.json # Full results across all models
├── final_cat_model.pkl          # Trained CatBoost model
├── final_lgbm_model.pkl         # Trained LightGBM model
├── final_rf_model.pkl           # Trained Random Forest model
├── final_xgb_model.pkl          # Trained XGBoost model
├── optuna_results.db            # Hyperparameter tuning database
├── requirements.txt             # Python dependencies
└── wals-v2020.4.zip             # WALS dataset archive
```

---

## 🧠 Models Used

| Model | Description |
|-------|-------------|
| **CatBoost** | Gradient boosting with categorical feature support |
| **LightGBM** | Fast gradient boosting framework |
| **XGBoost** | Extreme gradient boosting |
| **Random Forest** | Ensemble of decision trees |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/isharmi/Categorical-Bias-Multilingual.git
cd Categorical-Bias-Multilingual

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the main pipeline
```bash
python main.py
```

### Rebuild bias scores
```bash
python rebuild_bias_scores.py
```

---

## 📊 Dataset

This project uses the **World Atlas of Language Structures (WALS)** dataset (`wals-v2020.4`), which provides typological features across hundreds of languages. These features are used to analyze and predict bias scores in multilingual NLP systems.

---

## 📈 Results

Model results and evaluation metrics are stored in:
- `full_results_all_models.json` — full comparison across all models
- `bias_scores.csv` — per-language bias scores
- `results/` — detailed experiment outputs
- `evaluation/` — evaluation scripts and visualizations

---

## 🔧 Hyperparameter Tuning

Hyperparameter optimization is performed using **Optuna**. Tuning results are stored in:
- `optuna_results.db` — SQLite database of all trials
- `hyperparameter_tuning/` — tuning logs and configurations

---

## 📋 Requirements

See `requirements.txt` for the full list. Key dependencies include:

- `catboost`
- `lightgbm`
- `xgboost`
- `scikit-learn`
- `optuna`
- `pandas`
- `numpy`

---

## 👤 Author

**Sharmi Islam**
- GitHub: [@isharmi](https://github.com/isharmi)

---

## 📄 License

This project is for research purposes. Please cite appropriately if used in academic work.
