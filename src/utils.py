# src/utils.py
import os
import glob
import hashlib
import joblib
import json
import numpy as np
import pandas as pd
import shap
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# --- Basic paths (adjust if different) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

SHAP_GLOBAL_PATH = os.path.join(MODELS_DIR, "shap_global_summary.pkl")
SHAP_GLOBAL_PLOT = os.path.join(MODELS_DIR, "shap_global_plot.png")
SHAP_GLOBAL_BEESWARM = os.path.join(MODELS_DIR, "shap_global_beeswarm.png")
SHAP_GLOBAL_FULL = os.path.join(MODELS_DIR, "shap_global_full.npz")

# -------------------------
# Artifact loading helpers
# -------------------------
def find_latest_metadata(models_dir=MODELS_DIR):
    pattern = os.path.join(models_dir, "model_metadata_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def load_artifacts(models_dir=MODELS_DIR):
    """
    Load metadata JSON (latest) and pipelines if present.
    Returns a dict with keys:
    {
        'features', 'feature_types', 'cat_values',
        'calib_pipeline', 'raw_pipeline',
        'feature_means', 'metadata_path'
    }
    """
    artifacts = {
        "features": None,
        "feature_types": {},
        "cat_values": {},
        "calib_pipeline": None,
        "raw_pipeline": None,
        "feature_means": None,
        "metadata_path": None,
    }

    meta_path = find_latest_metadata(models_dir)
    feature_means_path = os.path.join(models_dir, "feature_means.json")
    if meta_path:
        artifacts["metadata_path"] = meta_path
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            # Load typical metadata keys
            artifacts["features"] = meta.get("features") or meta.get("feature_names") or meta.get("columns")
            artifacts["feature_types"] = meta.get("feature_types", {})
            artifacts["cat_values"] = meta.get("cat_values", {})

            # 🔹 Load feature means if available
            if os.path.exists(feature_means_path):
                try:
                    with open(feature_means_path, "r") as f:
                        artifacts["feature_means"] = json.load(f)
                except Exception:
                    artifacts["feature_means"] = {}
            else:
                artifacts["feature_means"] = {}

            # load calibrated/raw pipeline paths if present in metadata
            calib = meta.get("calibrated_pipeline") or meta.get("calibrated_model_path") or meta.get("calibrated_model")
            rawp = meta.get("raw_pipeline") or meta.get("raw_model_path") or meta.get("raw_pipeline_path")

            def _resolve(p):
                if not p:
                    return None
                if os.path.exists(p):
                    return p
                candidate = os.path.join(os.path.dirname(meta_path), os.path.basename(p))
                if os.path.exists(candidate):
                    return candidate
                candidate2 = os.path.join(models_dir, os.path.basename(p))
                if os.path.exists(candidate2):
                    return candidate2
                return None

            calibp = _resolve(calib)
            rawpp = _resolve(rawp)

            if rawpp:
                try:
                    artifacts["raw_pipeline"] = joblib.load(rawpp)
                except Exception:
                    artifacts["raw_pipeline"] = None
            if calibp:
                try:
                    artifacts["calib_pipeline"] = joblib.load(calibp)
                except Exception:
                    artifacts["calib_pipeline"] = None
        except Exception:
            pass

    # Fallback: load any joblib in models_dir with 'raw' or 'xgb' in filename
    if artifacts["raw_pipeline"] is None or artifacts["calib_pipeline"] is None:
        for f in os.listdir(models_dir):
            if f.endswith(".joblib") or f.endswith(".pkl"):
                if "raw" in f.lower() or "xgb" in f.lower() or "pipeline" in f.lower():
                    try:
                        obj = joblib.load(os.path.join(models_dir, f))
                        if isinstance(obj, Pipeline):
                            artifacts["raw_pipeline"] = artifacts["raw_pipeline"] or obj
                        else:
                            artifacts["calib_pipeline"] = artifacts["calib_pipeline"] or obj
                    except Exception:
                        continue
    return artifacts


# -------------------------
# Prediction helpers
# -------------------------
def model_predict_proba(pipe, X_df):
    """Return probability for positive class (works with pipelines and calibrated wrappers)."""
    if pipe is None:
        raise RuntimeError("No pipeline provided for prediction.")
    # if pipeline returns predict_proba directly
    try:
        proba = pipe.predict_proba(X_df)
        # assume binary
        if isinstance(proba, np.ndarray):
            if proba.ndim == 2:
                return proba[:, 1]
            else:
                return proba.ravel()
    except Exception:
        # try to extract estimator if calibrated
        if isinstance(pipe, Pipeline):
            last = list(pipe.named_steps.values())[-1]
            est = last
        else:
            est = pipe
        # If it's a calibrated wrapper
        if isinstance(est, CalibratedClassifierCV):
            base = getattr(est, "base_estimator_", None) or getattr(est, "estimator", None)
            if base is not None:
                try:
                    # run pipeline transform then estimator predict_proba
                    if isinstance(pipe, Pipeline):
                        # use pipeline without last step: get preprocessor
                        steps = list(pipe.named_steps.items())
                        if len(steps) >= 2:
                            # build transformed X by running everything except classifier
                            pre = Pipeline(list(steps[:-1]))
                            X_proc = pre.transform(X_df)
                            proba = base.predict_proba(X_proc)
                            return proba[:, 1]
                except Exception:
                    pass
        raise

def to_df_inputs(inputs: dict, features: list):
    """Convert dict of inputs (feature->value) into a single-row DataFrame with columns ordered as features."""
    # Ensure features exist
    row = {f: inputs.get(f, None) for f in features}
    df = pd.DataFrame([row], columns=features)
    return df

def risk_band_from_prob(p: float, low_thresh=0.4, high_thresh=0.7):
    if p >= high_thresh:
        return "High"
    if p >= low_thresh:
        return "Moderate"
    return "Low"

# -------------------------
# SHAP helpers (local & aggregation)
# --- paste/replace these three functions in src/utils.py ---

import re

def _normalize_name_for_matching(s):
    """Lowercase, replace non-alnum with underscore and strip repeated underscores."""
    if s is None:
        return ""
    s = str(s).lower()
    # convert unicode punctuation to underscore and keep alnum
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_{2,}', '_', s).strip('_')
    return s


def aggregate_shap_to_original(proc_feature_names, shap_array_1d, original_features):
    """
    Robust mapping from processed feature names -> original features with fallback heuristics.
    Returns: (agg_names, aggregated_signed_values)
    """
    proc = [str(p) for p in proc_feature_names]
    orig = [str(o) for o in original_features] if original_features is not None else []

    proc_norm = [_normalize_name_for_matching(p) for p in proc]
    orig_norm = [_normalize_name_for_matching(o) for o in orig]

    mapping = {}
    used_proc = set()

    # 1) exact normalized equality
    for i, pn in enumerate(proc_norm):
        for j, on in enumerate(orig_norm):
            if pn == on:
                mapping.setdefault(orig[j], []).append(i)
                used_proc.add(i)
                break

    # 2) normalized prefix or substring match
    for i, pn in enumerate(proc_norm):
        if i in used_proc:
            continue
        for j, on in enumerate(orig_norm):
            if pn.startswith(on) or on in pn:
                mapping.setdefault(orig[j], []).append(i)
                used_proc.add(i)
                break

    # 3) try heuristic matching where processed contains original token pieces
    for i, pn in enumerate(proc_norm):
        if i in used_proc:
            continue
        for j, on in enumerate(orig_norm):
            # compare splitted tokens
            pn_tokens = pn.split('_')
            on_tokens = on.split('_')
            common = set(pn_tokens).intersection(on_tokens)
            if len(common) >= 1:
                mapping.setdefault(orig[j], []).append(i)
                used_proc.add(i)
                break

    # 4) If nothing matched and counts look equal (rare), map by index as a fallback
    if len(mapping) == 0 and len(proc) == len(orig) and len(orig) > 0:
        for i in range(len(proc)):
            mapping.setdefault(orig[i], []).append(i)
            used_proc.add(i)

    # 5) Unmapped -> __other__
    unmapped = [i for i in range(len(proc)) if i not in used_proc]
    if unmapped:
        mapping.setdefault("__other__", []).extend(unmapped)

    # Build aggregated outputs
    agg_names = []
    agg_vals = []
    for o, idxs in mapping.items():
        vals = np.array(shap_array_1d)[idxs]
        agg_names.append(o)
        agg_vals.append(vals.sum())

    return agg_names, np.array(agg_vals)


def compute_local_shap_plot(raw_pipeline, X_row_df, original_features=None, top_k=12):
    """
    Compute a local SHAP figure aggregated to original features (best-effort).
    Returns matplotlib.figure
    """
    est = None
    pre = None
    if isinstance(raw_pipeline, Pipeline):
        # try typical names first
        if "preprocessor" in raw_pipeline.named_steps:
            pre = raw_pipeline.named_steps.get("preprocessor")
        else:
            # fallback: try to find a transformer-like step
            for name, step in raw_pipeline.named_steps.items():
                # heuristic: step with transform method but not the final estimator
                if name != list(raw_pipeline.named_steps.keys())[-1] and hasattr(step, "transform"):
                    pre = step
                    break
        last = list(raw_pipeline.named_steps.values())[-1]
        if isinstance(last, CalibratedClassifierCV):
            est = getattr(last, "base_estimator_", None) or getattr(last, "estimator", None) or last
        else:
            est = last
    else:
        est = raw_pipeline

    if est is None:
        raise RuntimeError("No estimator available for SHAP.")

    # prepare processed X and names
    if pre is not None:
        X_proc = pre.transform(X_row_df)
        if hasattr(X_proc, "toarray"):
            X_proc = X_proc.toarray()
        proc_feature_names = None
        try:
            # sklearn >= 1.0
            proc_feature_names = list(pre.get_feature_names_out(X_row_df.columns))
        except Exception:
            # fallback to length-based synthetic names
            if hasattr(X_proc, "shape"):
                proc_feature_names = [f"f{i}" for i in range(X_proc.shape[1])]
    else:
        X_proc = X_row_df.copy()
        proc_feature_names = list(X_proc.columns)

    # compute shap values
    try:
        try:
            expl = shap.TreeExplainer(est)
        except Exception:
            expl = shap.Explainer(est, X_proc)
        shap_vals = expl.shap_values(X_proc)
    except Exception as e:
        raise RuntimeError(f"SHAP explainer failed: {e}")

    if isinstance(shap_vals, list) and len(shap_vals) > 1:
        vals = np.array(shap_vals[1])
    else:
        vals = np.array(shap_vals)
    if vals.ndim == 2:
        vals_1d = vals[0]
    elif vals.ndim == 1:
        vals_1d = vals
    else:
        vals_1d = vals.reshape(vals.shape[0], -1)[0]

    # if original_features not provided, fallback to the incoming raw DataFrame columns
    if original_features:
        orig_feats = original_features
    else:
        orig_feats = list(X_row_df.columns)

    agg_names, agg_signed = aggregate_shap_to_original(proc_feature_names, vals_1d, orig_feats)

    order = np.argsort(np.abs(agg_signed))[::-1]
    top_order = order[:top_k]
    labels = [agg_names[i] for i in top_order][::-1]
    values = [float(agg_signed[i]) for i in top_order][::-1]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(labels) + 1)))
    colors = ["#2b8cbe" if v >= 0 else "#f03b20" for v in values]
    ax.barh(labels, values, color=colors)
    ax.set_title("Local SHAP contributions (top features)")
    ax.axvline(0, color="k", linewidth=0.6)
    plt.tight_layout()
    return fig


def get_local_shap_contributions(raw_pipeline, X_row_df, original_features=None, top_k=10):
    """
    Compute local SHAP contributions aggregated to original features and return
    a sorted list of (feature, signed_contribution, abs_contribution) sorted by abs desc.
    """
    import numpy as _np

    # reuse similar logic to compute_local_shap_plot but return numeric arrays
    est = None; pre = None
    if isinstance(raw_pipeline, Pipeline):
        # try to find preprocessor
        if "preprocessor" in raw_pipeline.named_steps:
            pre = raw_pipeline.named_steps.get("preprocessor")
        else:
            for name, step in raw_pipeline.named_steps.items():
                if name != list(raw_pipeline.named_steps.keys())[-1] and hasattr(step, "transform"):
                    pre = step
                    break
        last = list(raw_pipeline.named_steps.values())[-1]
        if isinstance(last, CalibratedClassifierCV):
            est = getattr(last, "base_estimator_", None) or getattr(last, "estimator", None) or last
        else:
            est = last
    else:
        est = raw_pipeline

    if est is None:
        raise RuntimeError("No estimator available for SHAP computation.")

    # prepare processed X and feature names
    if pre is not None:
        X_proc = pre.transform(X_row_df)
        if hasattr(X_proc, "toarray"):
            X_proc = X_proc.toarray()
        proc_feature_names = None
        try:
            proc_feature_names = list(pre.get_feature_names_out(X_row_df.columns))
        except Exception:
            if hasattr(X_proc, "shape"):
                proc_feature_names = [f"f{i}" for i in range(X_proc.shape[1])]
    else:
        X_proc = X_row_df.copy()
        proc_feature_names = list(X_proc.columns)

    # compute shap values
    try:
        try:
            expl = shap.TreeExplainer(est)
        except Exception:
            expl = shap.Explainer(est, X_proc)
        shap_vals = expl.shap_values(X_proc)
    except Exception as e:
        raise RuntimeError(f"SHAP explainer failed: {e}")

    if isinstance(shap_vals, list) and len(shap_vals) > 1:
        vals = _np.array(shap_vals[1])
    else:
        vals = _np.array(shap_vals)
    if vals.ndim == 2:
        vals_1d = vals[0]
    elif vals.ndim == 1:
        vals_1d = vals
    else:
        vals_1d = vals.reshape(vals.shape[0], -1)[0]

    # original features fallback
    if original_features:
        orig_feats = original_features
    else:
        orig_feats = list(X_row_df.columns)

    agg_names, agg_signed = aggregate_shap_to_original(proc_feature_names, vals_1d, orig_feats)

    abs_vals = _np.abs(agg_signed)
    order = _np.argsort(abs_vals)[::-1]
    result = []
    for idx in order[:top_k]:
        result.append({
            "feature": agg_names[idx],
            "signed": float(agg_signed[idx]),
            "abs": float(abs_vals[idx])
        })
    return result



def load_global_shap_summary(path=SHAP_GLOBAL_PATH):
    """Return summary dict if exists, else None"""
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        try:
            # some notebooks might save with numpy; try np.load
            import numpy as _np
            d = _np.load(path, allow_pickle=True)
            return dict(d)
        except Exception:
            return None


# -------------------------
# Batch prediction helper
# -------------------------

def run_batch_predictions(df, artifacts):
    """
    Run predictions on a DataFrame and return with added columns:
      - 'probability': predicted probability of positive class
      - 'Prediction': mapped risk band (Low/Moderate/High)
    """
    if artifacts.get("calib_pipeline") is not None:
        pipe = artifacts["calib_pipeline"]
    elif artifacts.get("raw_pipeline") is not None:
        pipe = artifacts["raw_pipeline"]
    else:
        raise RuntimeError("No valid pipeline found in artifacts.")

    # Probabilities
    probs = model_predict_proba(pipe, df)

    # Map to risk bands
    preds = [risk_band_from_prob(p) for p in probs]

    df_out = df.copy()
    df_out["probability"] = probs
    df_out["Prediction"] = preds
    return df_out

# -------------------------
# LLM Integration (Hugging Face API)
import requests

# --- Hugging Face Inference API helper ---
import time
import requests

def _format_shap_features(shap_df, top_n=5):
    """
    Convert shap dataframe to a readable list string:
      - expects shap_df with columns ['feature','mean_abs_shap','signed_shap'] or similar
    """
    if shap_df is None:
        return "No SHAP values available."

    # tolerant conversion
    try:
        if hasattr(shap_df, "sort_values"):
            top = shap_df.sort_values(by="mean_abs_shap", ascending=False).head(top_n)
            lines = []
            for _, r in top.iterrows():
                feat = r.get("feature") if "feature" in r.index else r.get("feature", "")
                mean_abs = float(r.get("mean_abs_shap", r.get("abs", 0)))
                signed = float(r.get("signed_shap", r.get("signed", 0)))
                direction = "increase" if signed > 0 else "decrease"
                lines.append(f"{feat} (impact {mean_abs:.3f}, tends to {direction} risk)")
            return "; ".join(lines) if lines else "No top features found."
        else:
            return str(shap_df)
    except Exception:
        return "Top contributing features not available."

# def _get_hf_token(st_module=None):
#     # prefer Streamlit secrets if available
#     try:
#         if st_module is not None and hasattr(st_module, "secrets") and "HF_API_TOKEN" in st_module.secrets:
#             return st_module.secrets["HF_API_TOKEN"]
#     except Exception:
#         pass
#     return os.environ.get("HF_API_TOKEN", None)


import os
import requests
import pandas as pd

# Primary and fallback models
HF_PRIMARY_URL = "https://api-inference.huggingface.co/models/Falconsai/medical_summarization"
HF_FALLBACK_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

def get_hf_token(st_module=None):
    """Load Hugging Face API token from Streamlit secrets or env variable."""
    try:
        if st_module is not None and hasattr(st_module, "secrets") and "HF_API_TOKEN" in st_module.secrets:
            return st_module.secrets["HF_API_TOKEN"]
    except Exception:
        pass
    return os.environ.get("HF_API_TOKEN", None)


def query_hf_model(prompt, api_url, token):
    """Helper: send request to Hugging Face Inference API."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300}}
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    # Handle different HF response formats
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    elif isinstance(data, list) and "summary_text" in data[0]:
        return data[0]["summary_text"].strip()
    elif isinstance(data, dict) and "summary_text" in data:
        return data["summary_text"].strip()
    else:
        return f"⚠️ Unexpected HF response: {data}"


def query_llm_summary(risk_band, probability, shap_df=None, top_n=6):
    """
    Query Hugging Face (medical_summarization → fallback to BART).
    Generates a clinician-friendly summary.
    """
    token = get_hf_token()
    if not token:
        return "⚠️ Hugging Face API token not found. Please set HF_API_TOKEN."

    # 🔹 Build features string
    prob_percent = f"{probability*100:.1f}%" if probability is not None else "N/A"
    features_text = ""
    if shap_df is not None and not shap_df.empty:
        top = shap_df.sort_values(by="mean_abs_shap", ascending=False).head(top_n)
        lines = []
        for _, r in top.iterrows():
            feat = r.get("feature", "")
            impact = r.get("mean_abs_shap", 0)
            signed = r.get("signed_shap", 0)
            direction = "increases" if signed > 0 else "decreases"
            lines.append(f"- {feat}: impact {impact:.3f}, tends to {direction} risk")
        features_text = "\n".join(lines)

    # 🔹 Clinical prompt
    prompt = f"""
    You are a clinical assistant AI.
    Based on the prediction results:

    - Risk Band: {risk_band}
    - Predicted Probability: {prob_percent}

    Top contributing features:
    {features_text if features_text else "No key features identified."}

    Write a short, clear clinical summary (2-3 sentences) for doctors.
    Include sections:
    💊 Medical Insight  
    🥦 Diet Tips  
    🏃 Lifestyle Suggestions
    """

    # 🔹 Try primary model first
    try:
        return query_hf_model(prompt, HF_PRIMARY_URL, token)
    except Exception as e1:
        print(f"⚠️ Primary model failed: {e1}, falling back to BART...")
        try:
            return query_hf_model(prompt, HF_FALLBACK_URL, token)
        except Exception as e2:
            return f"⚠️ Hugging Face API error (both models failed): {e2}"


def format_summary_text(summary_text: str) -> str:
    """
    Format the raw model output into structured markdown sections.
    Ensures headings for Medical, Diet, and Lifestyle.
    """
    if not summary_text or summary_text.startswith("⚠️"):
        return summary_text

    # Normalize text
    text = summary_text.strip()

    # Try to detect existing sections
    has_medical = "medical" in text.lower()
    has_diet = "diet" in text.lower()
    has_lifestyle = "lifestyle" in text.lower()

    formatted = []

    # Force structure if missing
    if has_medical:
        formatted.append("💊 **Medical Insight**\n" + extract_section(text, "medical"))
    else:
        formatted.append("💊 **Medical Insight**\n" + text.split(".")[0] + ".")

    if has_diet:
        formatted.append("🥦 **Diet Tips**\n" + extract_section(text, "diet"))
    else:
        formatted.append("🥦 **Diet Tips**\n- Eat a balanced diet with fruits, vegetables, and whole grains.\n- Limit fried and processed foods.")

    if has_lifestyle:
        formatted.append("🏃 **Lifestyle Suggestions**\n" + extract_section(text, "lifestyle"))
    else:
        formatted.append("🏃 **Lifestyle Suggestions**\n- Stay active with regular walking or exercise.\n- Maintain healthy body weight.")

    return "\n\n".join(formatted)


def extract_section(text: str, keyword: str) -> str:
    """
    Extracts a section of text around a keyword (e.g., medical/diet/lifestyle).
    If not found cleanly, returns a placeholder.
    """
    lines = text.split("\n")
    section_lines = []
    capture = False
    for line in lines:
        if keyword.lower() in line.lower():
            capture = True
        elif capture and (any(k in line.lower() for k in ["medical", "diet", "lifestyle"])):
            break
        if capture:
            section_lines.append(line)
    return "\n".join(section_lines).strip() if section_lines else "No specific advice provided."

def top_features_explanation(shap_df, top_n=5):
    """
    Generate short explanations for top contributing features.
    """
    if shap_df is None or shap_df.empty:
        return "No top features available."

    top = shap_df.sort_values(by="mean_abs_shap", ascending=False).head(top_n)
    lines = []
    for _, r in top.iterrows():
        feat = r.get("feature", "")
        signed = r.get("signed_shap", 0)
        direction = "increases" if signed > 0 else "reduces"
        lines.append(f"- **{feat}**: tends to {direction} gallstone risk.")
    return "\n".join(lines)
