# Loan Risk Management System (ML)

This repository packages the Streamlit demonstration app and project instructions for the Loan Risk Management ML project.

## Project files

- `app.py` — Streamlit user interface and prediction code.
- `requirements.txt` — Python dependencies.
- `models/` — add the trained model and fitted preprocessing object here.
- `.streamlit/config.toml` — Streamlit server configuration.

## Required model files

The trained `.pkl` files are not embedded in this ZIP because they were created in your Google Colab runtime and were not provided as downloadable attachments in this chat. Download them from the same Colab runtime:

```python
from google.colab import files
files.download("/content/loan_risk_model.pkl")
files.download("/content/loan_risk_preprocessor.pkl")
```

Put both downloaded files into the `models/` folder without renaming:

```text
models/loan_risk_model.pkl
models/loan_risk_preprocessor.pkl
```

The app cannot make predictions until these files are present.

## Install and run locally

Python 3.10 or 3.11 is recommended. In a terminal opened in this project folder:

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## GitHub upload

1. Extract this ZIP.
2. Add the two `.pkl` files to `models/`.
3. Create a new GitHub repository.
4. Upload the extracted project files and folders.
5. Commit the files.

Pickle files can execute code when loaded. Only use model files you created and trust. If they are large, use Git LFS or private artifact storage. Do not commit secrets or API keys.

## Dataset files used in the Colab workflow

- `traindemographics.csv`
- `trainperf.csv`
- `trainpreviousloans.csv`

The CSVs are not included in this ZIP. Add them only if the dataset license permits redistribution. Before publishing, add the original Kaggle dataset URL and attribution here.

## Project workflow

1. Load demographic, current-loan performance, and previous-loan CSV files.
2. Inspect missing values and duplicate customer records.
3. Aggregate previous-loan information by customer.
4. Merge datasets and engineer loan/customer features.
5. Prepare numerical and categorical columns.
6. Split into training and test sets.
7. Fit preprocessing on training data and apply SMOTE to training data only.
8. Train Logistic Regression, Random Forest, and XGBoost.
9. Compare accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and ROC curves.
10. Save the selected Logistic Regression model and fitted preprocessor.

## Reported initial model results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5767 | 0.2877 | 0.6421 | 0.3974 | 0.6336 |
| Random Forest | 0.7300 | 0.3403 | 0.2579 | 0.2934 | 0.6370 |
| XGBoost | 0.7494 | 0.3165 | 0.1316 | 0.1859 | 0.6089 |

Logistic Regression was selected in the notebook because it had the highest Bad-class recall and F1-score among these three initial runs. These are preliminary holdout results, not proof of real-world lending performance.

## Limitations

- This is an academic prototype, not a validated credit-scoring product.
- The UI is a simplified form. Its raw input mapping must match the exact columns/categories used when fitting the saved preprocessor. If the notebook used additional raw columns or different labels, update `row` in `app.py` accordingly.
- Model probabilities may not be calibrated.
- Do not use this model as the sole basis for real lending decisions.
