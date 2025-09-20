# pages/batch.py
import streamlit as st
import pandas as pd
from io import BytesIO
from src.utils import (
    run_batch_predictions,
    compute_local_shap_plot,
    get_local_shap_contributions,
)

def app(artifacts):
    st.title("📂 Batch Prediction")

    st.markdown(
        "Upload a CSV file with patient data. Predictions will be generated using the trained model. "
        "You can also explore **Local SHAP explanations** for each patient row."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.success(f"✅ Loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns.")
        st.write("### Preview of uploaded data")
        st.dataframe(df.head())

        # --- Run predictions ---
        if st.button("Run Predictions"):
            with st.spinner("Running predictions..."):
                try:
                    df_preds = run_batch_predictions(df, artifacts)
                    st.session_state["batch_predictions"] = df_preds
                    st.success("✅ Predictions generated!")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        # --- Show predictions if available ---
        if "batch_predictions" in st.session_state:
            df_preds = st.session_state["batch_predictions"]

            st.write("### Predictions (first 20 rows)")
            st.dataframe(df_preds.head(20))

            # Download predictions
            csv_buffer = BytesIO()
            df_preds.to_csv(csv_buffer, index=False)
            st.download_button(
                "⬇️ Download Predictions (CSV)",
                data=csv_buffer.getvalue(),
                file_name="batch_predictions.csv",
                mime="text/csv",
            )

            # --- Local SHAP for a selected patient ---
            st.markdown("## 🧑‍⚕️ Local SHAP Explanations (for CSV patients)")

            row_index = st.number_input(
                "Select Patient Row Index",
                min_value=0,
                max_value=len(df_preds) - 1,
                step=1,
                value=0,
            )

            # Use the same data as predictions, but remove target columns
            X_row = df_preds.drop(
                columns=["probability", "Prediction"], errors="ignore"
            ).iloc[[row_index]]

            raw_pipe = artifacts.get("raw_pipeline")

            feature_names = artifacts.get("features")
            if not feature_names:
                feature_names = list(X_row.columns)

            if raw_pipe is not None:
                try:
                    # --- Styled prediction card for selected row ---
                    prob = df_preds.iloc[row_index]["probability"]
                    band = df_preds.iloc[row_index]["Prediction"]
                    prob_percent = prob * 100

                    if band == "High":
                        color = "red"
                    elif band == "Moderate":
                        color = "orange"
                    else:
                        color = "green"

                    st.markdown(
                        f"""
                        <div style="background-color:#ffffff; padding:20px; border-radius:12px; 
                                    border: 2px solid {color}; box-shadow: 0px 4px 10px rgba(0,0,0,0.15); 
                                    text-align:center; margin-top:10px;">
                            <h3 style="color:{color}; margin-bottom:15px;">Prediction Result: {band} Risk</h3>
                            <p style="font-size:18px; color:#333; margin:0;">
                                <b>Risk Percentage:</b> 
                                <span style="color:{color}; font-weight:bold;">{prob_percent:.1f}%</span>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # SHAP plot
                    fig = compute_local_shap_plot(
                        raw_pipe,
                        X_row,
                        original_features=feature_names,
                        top_k=12,
                    )
                    st.pyplot(fig)

                    # Tabular SHAP contributions
                    contribs = get_local_shap_contributions(
                        raw_pipe,
                        X_row,
                        original_features=feature_names,
                        top_k=10,
                    )
                    if contribs:
                        shap_df = pd.DataFrame(contribs)
                        st.markdown("**Local SHAP Contributions (Top Features)**")
                        st.dataframe(shap_df)

                        # Download SHAP CSV
                        shap_csv = shap_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download Local SHAP Contributions (CSV)",
                            data=shap_csv,
                            file_name=f"local_shap_row{row_index}.csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    st.error(f"Local SHAP failed: {e}")
            else:
                st.info("Local SHAP not available (raw pipeline missing).")
