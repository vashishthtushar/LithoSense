# pages/explainability.py
import os
import streamlit as st
import pandas as pd

def app(artifacts):
    st.title("🌍 Explainability")

    MODELS_DIR = "models"
    bar_path = os.path.join(MODELS_DIR, "shap_global_bar.png")
    beeswarm_path = os.path.join(MODELS_DIR, "shap_global_beeswarm.png")
    csv_path = os.path.join(MODELS_DIR, "shap_feature_importance.csv")

    # --- Global SHAP Bar Plot ---
    st.subheader("Global SHAP Bar Plot")
    if os.path.exists(bar_path):
        st.image(bar_path, caption="Global SHAP Bar Plot", use_column_width=True)
    else:
        st.warning("Global SHAP bar plot not found in models/")

    # --- Global SHAP Beeswarm Plot ---
    st.subheader("Global SHAP Beeswarm Plot")
    if os.path.exists(beeswarm_path):
        st.image(beeswarm_path, caption="Global SHAP Beeswarm Plot", use_column_width=True)
    else:
        st.warning("Global SHAP beeswarm plot not found in models/")

    # --- Global SHAP Summary Table ---
    st.subheader("Global SHAP Summary Table")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            st.dataframe(df.head(20))  # show top 20 features

            with open(csv_path, "rb") as f:
                st.download_button(
                    label="📥 Download Full SHAP Summary (CSV)",
                    data=f,
                    file_name="shap_feature_importance.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Could not load SHAP summary CSV: {e}")
    else:
        st.info("SHAP summary CSV not found in models/")
