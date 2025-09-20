import streamlit as st
import os
import base64

def app(artifacts):
    logo_path = "Gemini_Generated_Image_gi8m9lgi8m9lgi8m.png"

    def get_base64_image(img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    if os.path.exists(logo_path):
        logo_base64 = get_base64_image(logo_path)
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: -10px;">
                <img src="data:image/png;base64,{logo_base64}" 
                     style="width:110px; border-radius:50%; 
                            box-shadow: 0px 0px 10px rgba(0,0,0,0.3);" />
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<h1 style='text-align:center; color:#2E7D32;'>LithoSense</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#90CAF9;'>Gallstone Risk Prediction & Clinical Insights</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # Welcome block
    st.markdown("""
    ### 👋 Welcome to LithoSense  
    LithoSense is a **clinical decision-support tool** to help assess gallstone risk.  
    It combines **AI-powered predictions** with **transparent explanations**, supporting healthcare professionals in clinical decision-making.  
    """)

    # Key Features styled
    st.markdown("""
    <div style="background-color:#1E1E1E; padding:20px; border-radius:12px; margin-bottom:20px;">
        <h4 style="color:#4CAF50;">Key Features</h4>
        <ul style="line-height:1.8; font-size:16px;">
            <li>👤 <b>Single Patient Prediction</b> — Enter patient details and get risk assessment.</li>
            <li>📊 <b>Batch Predictions</b> — Upload CSV for multiple patients.</li>
            <li>🧪 <b>Explainability</b> — Visualize <i>why</i> the model made predictions (SHAP plots).</li>
            <li>⚖️ <b>Fairness Analysis</b> — Ensure risk predictions are balanced across patient groups.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧾 Model Information")
    md = artifacts.get("metadata_path") or "Not available"
    has_calib = "✅ Available" if artifacts.get("calib_pipeline") is not None else "❌ Not available"
    has_raw = "✅ Available" if artifacts.get("raw_pipeline") is not None else "❌ Not available"

    st.markdown(f"- **Metadata file:** `{md}`")
    st.markdown(f"- **Calibrated model:** {has_calib}")
    st.markdown(f"- **Raw model:** {has_raw}")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; font-size:16px; color:#B0BEC5;'>"
        "➡️ Use the left sidebar to start exploring predictions and explanations."
        "</div>", 
        unsafe_allow_html=True
    )
