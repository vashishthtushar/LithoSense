# pages/fairness.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def app(artifacts):
    st.title("⚖️ Fairness & Risk Comparison")

    if "batch_predictions" not in st.session_state or st.session_state["batch_predictions"] is None:
        st.info("Please run Batch Predictions first to view Fairness results.")
        return

    df = st.session_state["batch_predictions"].copy()
    df.columns = df.columns.str.strip()

    required_cols = ["Prediction", "probability", "Gender", "Age", "Comorbidity"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.warning("Some important patient details are missing, fairness check skipped.")
        return

    st.subheader("Risk Comparison Across Patient Groups")

    # Create Age bins
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 40, 60, 100],
        labels=["<40", "40–59", "60+"],
        right=False
    )

    # Define subgroup labels
    subgroups = {
        "Gender": {0: "Male", 1: "Female"},
        "AgeGroup": {"<40": "<40", "40–59": "40–59", "60+": "60+"},
        "Comorbidity": {0: "No Comorbidity", 1: "With Comorbidity"}
    }

    overall_avg = df["probability"].mean() * 100  # convert to %

    for feature, mapping in subgroups.items():
        st.markdown(f"### 📊 {feature} Groups")

        df_display = df.copy()
        if feature in mapping:
            df_display[feature] = df_display[feature].map(mapping).fillna(df_display[feature])

        # Compute subgroup averages
        summary = df_display.groupby(feature).agg(
            avg_risk=("probability", lambda x: x.mean() * 100),
            high_risk_rate=("Prediction", lambda x: (x == "High").mean() * 100),
            count=("Prediction", "count")
        ).reset_index()

        # Show clean summary table
        summary.rename(
            columns={
                "avg_risk": "Average Risk %",
                "high_risk_rate": "High Risk %",
                "count": "Number of Patients"
            },
            inplace=True
        )
        st.dataframe(summary, use_container_width=True)

        # Show bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(summary[feature], summary["Average Risk %"], color="skyblue")
        ax.axhline(overall_avg, color="red", linestyle="--", label="Overall Avg")
        ax.set_title(f"Average Risk % by {feature}")
        ax.set_ylabel("Average Risk %")
        ax.legend()
        st.pyplot(fig)

        # Show easy interpretation
        for _, row in summary.iterrows():
            st.markdown(
                f"- **{feature}: {row[feature]}** → Average Risk: "
                f"<span style='color:blue; font-weight:bold;'>{row['Average Risk %']:.1f}%</span> "
                f"({int(row['Number of Patients'])} patients)",
                unsafe_allow_html=True
            )

    st.success("✅ Fairness check completed. These results show how risk differs across patient groups.")
