# pages/single_patient.py
import streamlit as st
import pandas as pd
from src.utils import model_predict_proba, compute_local_shap_plot, get_local_shap_contributions, query_llm_summary
import hashlib
from textwrap import dedent

def app(artifacts):
    st.markdown("## 👤 Single Patient Prediction")

    # 📌 Styled note
    st.markdown(
        """
        <div style="background-color:#FFF3CD; padding:12px; border-radius:8px; 
                    border:1px solid #FFEEBA; color:#856404; font-size:15px; margin-bottom:15px;">
            <b>Note:</b> For selecting <b>Gender</b>, use 
            <b style="color:#d9534f;">0</b> for Female and 
            <b style="color:#0275d8;">1</b> for Male.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Enter Patient Data")

    feature_names = artifacts.get("features", [])
    feature_means = artifacts.get("feature_means", {})

    # if not feature_names:
    #     st.error("No feature names found in artifacts.")
    #     return

    # Initialize defaults only once
    if "initialized_defaults" not in st.session_state:
        for feature in feature_names:
            try:
                st.session_state[f"input_{feature}"] = float(
                    feature_means.get(feature, 0.0)
                )
            except Exception:
                st.session_state[f"input_{feature}"] = 0.0
    
        st.session_state["initialized_defaults"] = True

    
    # Multi-column form
    with st.form("single_patient_form"):
        inputs = {}
        n_cols = 3
        rows = (len(feature_names) + n_cols - 1) // n_cols

        for r in range(rows):
            cols = st.columns(n_cols)
            for i, col in enumerate(cols):
                idx = r * n_cols + i
                if idx < len(feature_names):
                    feature = feature_names[idx]
                    default_val = float(feature_means.get(feature, 0.0))  # ✅ use means
                    inputs[feature] = col.number_input(
                        feature,
                        # value=default_val,
                        step=0.01,
                        format="%.4f",
                        key=f"input_{feature}"
                    )

        submitted = st.form_submit_button("🔍 Run Prediction")

    if submitted:
        X = pd.DataFrame([inputs])
        pipe = artifacts.get("calib_pipeline") or artifacts.get("raw_pipeline")

        if pipe is None:
            st.error("No model pipeline found in artifacts.")
            return

        # Predict probability
        prob = model_predict_proba(pipe, X)[0]

        # Risk band
        if prob > 0.66:
            risk_band, color = "High Risk", "red"
        elif prob > 0.33:
            risk_band, color = "Moderate Risk", "orange"
        else:
            risk_band, color = "Low Risk", "green"

        # Tabs for results
        tab1, tab2, tab3 = st.tabs(["🧾 Risk Card", "📊 Local SHAP", "🤖 AI-Generated Patient Summary"])

        with tab1:
            # Convert to percentage
            prob_percent = prob * 100

            # Risk Card styling
            st.markdown(
                f"""
                <div style="background-color:#ffffff; padding:20px; border-radius:12px; 
                            border: 2px solid {color}; box-shadow: 0px 4px 10px rgba(0,0,0,0.15); 
                            text-align:center; margin-top:10px;">
                    <h2 style="color:{color}; margin-bottom:15px;">🧾 Prediction Result: {risk_band}</h2>
                    <p style="font-size:18px; color:#333; margin:0;">
                        <b>Risk Percentage:</b> 
                        <span style="color:{color}; font-weight:bold;">{prob_percent:.1f}%</span>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Save prediction result into session state
            st.session_state["prediction_result"] = {
                "risk": risk_band,
                "probability": prob
            }
            # Save patient input values so AI tab can use them later
            st.session_state["patient_values"] = inputs

            # Invalidate old summary when new prediction is made
            if "ai_summary" in st.session_state:
                del st.session_state["ai_summary"]



        with tab2:
            st.markdown("### Local SHAP Explanation")
            raw_pipe = artifacts.get("raw_pipeline")
            if raw_pipe is not None:
                try:
                    fig = compute_local_shap_plot(
                        raw_pipe, X, original_features=feature_names, top_k=12
                    )
                    st.pyplot(fig)
        
                    contribs = get_local_shap_contributions(
                        raw_pipe, X, original_features=feature_names, top_k=8
                    )
                    if contribs:
                        # convert to df and rename keys to what AI tab expects
                        shap_df = pd.DataFrame(contribs)
                        # rename 'abs' -> 'mean_abs_shap', 'signed' -> 'signed_shap'
                        rename_map = {}
                        if 'abs' in shap_df.columns:
                            rename_map['abs'] = 'mean_abs_shap'
                        if 'signed' in shap_df.columns:
                            rename_map['signed'] = 'signed_shap'
                        if rename_map:
                            shap_df = shap_df.rename(columns=rename_map)

                        st.markdown("**Top Feature Contributions**")
                        st.dataframe(shap_df)
        
                        # ✅ Save SHAP results for AI summary tab (use DataFrame with expected column names)
                        st.session_state["shap_values"] = shap_df
        
                except Exception as e:
                    st.error(f"Local SHAP failed: {e}")
            else:
                st.info("Local SHAP not available (raw pipeline missing).")

            # Invalidate old summary when new SHAP is computed
            if "ai_summary" in st.session_state:
                del st.session_state["ai_summary"]

        with tab3:
            st.header("🤖 AI-Generated Patient Summary")
            st.info("AI-generated summary based on Risk Card and Local SHAP results.")
        
            if "prediction_result" not in st.session_state:
                st.warning("⚠️ Please run the Risk Card prediction first.")
            else:
                pred = st.session_state["prediction_result"]
                risk_band = pred.get("risk", "Unknown")
                probability = pred.get("probability", None)
        
                shap_vals = st.session_state.get("shap_values", None)
        
                if shap_vals is None:
                    st.warning("⚠️ Please generate Local SHAP explanation first.")
                else:
                    from src.utils import format_summary_text

                    summary_text = query_llm_summary(risk_band, probability, shap_vals, top_n=6)
                    if summary_text.startswith("⚠️"):
                        st.error(summary_text)
                    else:
                        st.success("✅ Summary generated")
                        formatted = format_summary_text(summary_text)
                        st.markdown(formatted)

                    from src.utils import top_features_explanation

                    # Show Top 5 Feature Explanations
                    st.markdown("### 🔎 Top 5 Contributing Features")
                    feature_summary = top_features_explanation(shap_vals, top_n=5)
                    st.markdown(feature_summary)





    

        
    


