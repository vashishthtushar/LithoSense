# LithoSense — Clinical Gallstone Risk Screening (Interactive + Explainable)

**Author:** Tushar  
**Status:** Prototype / Demo-ready (Streamlit) — includes prediction, explainability (SHAP), fairness checks, and LLM-based clinical summaries (Hugging Face / fallback).

> ⚠️ **Medical disclaimer:** This project is an academic prototype. It must **not** be used as a substitute for professional medical diagnosis or treatment. All outputs should be reviewed by qualified clinicians before any patient-facing use.

---

## Table of contents
- [Project overview](#project-overview)  
- [Features](#features)  
- [Repository layout](#repository-layout)  
- [Quick start (local)](#quick-start-local)  
- [Environment & requirements](#environment--requirements)  
- [Secrets (LLM API keys)](#secrets-llm-api-keys)  
- [Run the app locally](#run-the-app-locally)  
- [Deploy (Streamlit Cloud / Hugging Face Spaces)](#deploy-streamlit-cloud--hugging-face-spaces)  
- [How LLM integration works](#how-llm-integration-works)  
- [Explainability report contents (for clinicians)](#explainability-report-contents-for-clinicians)  
- [Interactive clinical screening UI walkthrough](#interactive-clinical-screening-ui-walkthrough)  
- [How to retrain / reproduce baseline](#how-to-retrain--reproduce-baseline)  
- [Evaluation & metrics](#evaluation--metrics)  
- [Debugging tips & common issues](#debugging-tips--common-issues)  
- [Future work](#future-work)  
- [Contributing](#contributing)  
- [License & acknowledgements](#license--acknowledgements)

---

## Project overview
LithoSense is an interactive clinical screening tool that predicts gallstone risk from patient features, explains the prediction using SHAP, and generates clinician-friendly summaries using an integrated LLM. The app is built with Streamlit and designed for non-technical clinical users.

Primary goals:
- Provide a simple UI for single-patient and batch screening.
- Surface the most influential features that drove the model prediction (with visuals).
- Produce short clinician-friendly clinical summaries and action-oriented advice (diet/lifestyle) using an LLM API.
- Provide an explainability report suitable for clinical audiences.

---

## Features
- Single-patient input form → risk score (probability + categorical risk level).
- Batch CSV upload for screening multiple patients.
- Local SHAP explainability (summary plot and per-patient contributions).
- Top-5 features with one-sentence explanations.
- LLM-generated clinical summary (Medical Insight, Diet Tips, Lifestyle Suggestions).
- Fairness checks / basic demographic analysis page.
- Configurable LLM backend (Hugging Face inference with model fallbacks; optional OpenAI).

---

## Repository layout (expected)
```
LithoSense/
├── app.py                       # Streamlit entrypoint (routes to pages)
├── requirements.txt
├── .streamlit/
│   └── secrets.toml             # DO NOT COMMIT: contains HF/OPENAI keys
├── _pages/                      # Streamlit multipage scripts
│   ├── single_patient.py
│   ├── batch.py
│   ├── explainability.py
│   └── fairness.py
├── src/
│   └── utils.py                 # Utilities: model load, predict, SHAP, LLM helpers
├── models/                      # Saved model artifacts (state_dict / joblib / pickle)
├── plots/                       # saved explainability images (optional)
├── data/                        # sample CSVs, test cases
├── notebooks/                    # EDA / training / model_zoo / explainability notebooks
└── README.md
```

---

## Quick start (local)

1. **Clone repo**
   ```bash
   git clone https://github.com/<your-username>/LithoSense.git
   cd LithoSense
   ```

2. **Create environment (conda recommended)**
   ```bash
   conda create -n lithosense python=3.10 -y
   conda activate lithosense
   ```

3. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

---

## Environment & requirements
Minimum `requirements.txt` (already present in the repo):

```text
streamlit
scikit-learn
pandas
numpy
joblib
shap
matplotlib
seaborn
xgboost
lightgbm
catboost
openpyxl
pyyaml
requests
orjson
```

> If you are not using some model packages (catboost/lightgbm/xgboost), you can remove them to reduce install time.

---

## Secrets (LLM API keys)
For LLM integration, the app looks for the Hugging Face token (`HF_API_TOKEN`) in Streamlit secrets or environment variable. Example `.streamlit/secrets.toml` (LOCAL ONLY — do NOT commit):

```toml
HF_API_TOKEN = "hf_xxx_your_token_here"
# Optional: OPENAI key if you want to switch
# OPENAI_API_KEY = "sk-..."
```

**Important**
- Never commit `secrets.toml` to GitHub.
- If a secret accidentally got committed, rotate the token immediately (Hugging Face → Settings → Access Tokens), and scrub it from history (use `git-filter-repo` or fresh clone & force push) — instructions exist in this repo's docs/notes.

---

## Run the app locally

1. Ensure the conda env is activated and secrets are set.
2. Start Streamlit:
   ```bash
   streamlit run app.py
   ```
3. Open browser at `http://localhost:8501`.

**Notes**
- If LLM responses are slow or failing, check `HF_API_TOKEN` and the model endpoint (rate limits / model gating).
- For faster local development without LLM, comment-out or bypass the LLM call (the UI still shows SHAP results & risk card).

---

## Deploy (Streamlit Cloud / Hugging Face Spaces)

### Streamlit Community Cloud (recommended, fastest)
1. Push your repo to GitHub (do not include `.streamlit/secrets.toml`). Use `.gitignore`.
2. Go to https://share.streamlit.io → New app → connect to GitHub → choose repo & `app.py`.
3. In the app settings → Secrets → add `HF_API_TOKEN` (and `OPENAI_API_KEY` if used).
4. Deploy and use the public URL.

### Hugging Face Spaces
1. Create a new Space (type: Streamlit).
2. Connect the repository or upload code.
3. Add `requirements.txt`.
4. Set `HF_API_TOKEN` in Space settings (Secrets).
5. Launch.

---

## How LLM integration works

**Design**
- LLM is used only to generate clinician-friendly summaries and insights from model output + SHAP top features.
- The LLM helper is in `src/utils.py` (function: `query_llm_summary(...)`).
- Implementation tries primary domain model (e.g., `Falconsai/medical_summarization`) and falls back to `facebook/bart-large-cnn` if access issues occur.

**Prompting**
- The tool sends a short contextual prompt including:
  - Risk band (Low/Moderate/High)
  - Probability (%)
  - Top features (top-5 from SHAP, with direction)
- The desired output format is forced: three sections — `💊 Medical Insight`, `🥦 Diet Tips`, `🏃 Lifestyle` (bullet lists).

**Switching to OpenAI (optional)**
- If you have OpenAI credits, you can switch to `gpt-3.5-turbo` by setting `OPENAI_API_KEY` in secrets and switching the `query_llm_summary` implementation to call OpenAI. See `src/utils.py` comments for example code.

---

## Explainability report contents (for clinicians)
The Explainability report produced by the app is intentionally **non-technical** and oriented toward clinicians:

1. **Risk score & band** — Probability and categorical band (Low / Moderate / High), with short interpretation.
2. **Top influential factors** — Top-5 features from SHAP with one-line interpretation of direction & relative impact.
3. **SHAP visualizations**
   - Global SHAP summary plot (feature importance).
   - Local SHAP plot for the individual patient (waterfall or bar).
4. **Actionable recommendations** — Short diet & lifestyle suggestions generated by LLM (3–5 bullets).
5. **Limitations & caveats** — Notes on model scope, population used for training, calibration, and the need for clinical judgement.

---

## Interactive clinical screening UI walkthrough
- **Home**: Project description & instructions.
- **Single Patient**: Input patient features, click *Run Prediction*, view risk card, view Local SHAP, then AI Summary (auto-generated or on-button depending on configuration).
- **Batch CSV**: Upload a CSV with required columns, download results with risk column appended.
- **Explainability**: Full report, downloadable figures for clinician notes.
- **Fairness**: Simple subgroup performance and representation checks.

---

## How to retrain / reproduce baseline
Notebooks (in `notebooks/`):
- `00_eda.ipynb` — Exploratory data analysis
- `10_preprocess.ipynb` — Feature engineering & preprocessing pipeline
- `20_model_zoo.ipynb` — Training baseline models and comparison (XGBoost / RandomForest / LightGBM)
- `30_explainability.ipynb` — SHAP experiments and plots

General steps:
1. Prepare raw dataset in `data/` (ensure column names match UI fields).
2. Run preprocessing notebook to produce features and train/test split.
3. Train model and save artifact into `models/` (e.g., `models/model.joblib`).
4. Update `src/utils.py` to point to the saved model if path differs and restart app.

---

## Evaluation & metrics
Use standard classification metrics:
- AUC-ROC (primary), Accuracy, Precision, Recall, F1-score, Calibration plots.
- Confusion matrices for clinically relevant thresholds.
- Reporting should include subgroup metrics (age/gender/comorbidity) for fairness.

---

## Debugging tips & common issues
- **LLM returns errors (403/404)**: model is gated or token lacks inference permission. Use a public model or re-generate token with proper scope.
- **LLM returns 429**: rate-limited or insufficient quota. Use fallback model or add backoff/retry.
- **Secrets accidentally committed**: rotate token immediately and scrub commit history (see GitHub docs / `git-filter-repo`). Do not push secrets.
- **Streamlit shows no updates after changing secrets**: restart Streamlit process.
- **SHAP fails**: ensure model and input features shapes align; convert shap arrays to DataFrame with `feature` and `signed_shap` columns.

---

## Future work / extension ideas
- Host a dedicated inference endpoint for a more powerful instruction-tuned LLM (Meta-LLaMA, Mistral) in a controlled environment.
- Add user authentication & audit logs (for production clinical settings).
- Fine-tune a domain-specific LLM (if allowed) on clinical summaries and model rationale.
- Add model monitoring (drift detection) and more advanced fairness audits.
- Add multi-language support for summaries.

---

## Contributing
If you want to contribute:
1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Make changes, commit, and open a pull request with a clear description.
4. Keep secrets out of the repo.

---

## License
Suggested: **MIT License** — include `LICENSE` in the repo.  
(If you prefer another license, add the appropriate file.)

---

## Acknowledgements & references
- SHAP: Lundberg et al., for explainability methods.
- Hugging Face / Transformers for LLM access.
- Streamlit for the app UI.

---

## Contact
If you want help integrating additional features (model deployment, LLM tuning, production infra), say the word — I can provide deployment scripts, Dockerfiles, or a step-by-step guide for hosting on an inference endpoint or cloud VM.
