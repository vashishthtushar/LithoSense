import streamlit as st

st.set_page_config(page_title="LithoSense", layout="wide")

# Import your app pages
from _pages import home, single_patient, batch, explainability, fairness

# Import utils
from src.utils import load_artifacts

# ------------------------------
# Main app
# ------------------------------
def main():
    # st.set_page_config(page_title="LithoSense", layout="wide")

    # ✅ Load artifacts once here
    artifacts = load_artifacts()

    st.sidebar.title("🔎 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "👤 Single Patient", "📂 Batch CSV (upload)", "📊 Explainability", "⚖ Fairness"],
        index=0,
    )

    # Route pages
    if page == "🏠 Home":
        home.app(artifacts)
    elif page == "👤 Single Patient":
        single_patient.app(artifacts)
    elif page == "📂 Batch CSV (upload)":
        batch.app(artifacts)
    elif page == "📊 Explainability":
        explainability.app(artifacts)
    elif page == "⚖ Fairness":
        fairness.app(artifacts)


if __name__ == "__main__":
    main()
