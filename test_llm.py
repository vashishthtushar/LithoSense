from src.utils import query_local_llm

prompt = "Summarize: A patient has high cholesterol and obesity, what are contributing factors and lifestyle tips?"
print("Querying local LLM...")
print(query_local_llm(prompt, max_tokens=200))




# import streamlit as st
# from src.utils import load_artifacts
# from src.utils import query_local_llm

# # import page modules from renamed folder
# from _pages import home, single_patient, batch, explainability, fairness

# def main():
#     st.set_page_config(
#         layout="wide",
#         page_title="LithoSense",
#         page_icon="🪨",
#         initial_sidebar_state="expanded"
#     )

#     artifacts = load_artifacts()

#     # Sidebar Navigation
#     st.sidebar.markdown("## 🔍 Navigation")

#     menu = {
#         "🏠 Home": home,
#         "👤 Single Patient": single_patient,
#         "📂 Batch CSV (Upload)": batch,
#         "🧪 Explainability": explainability,
#         "⚖️ Fairness": fairness,
#     }

#     choice = st.sidebar.radio("Go to", list(menu.keys()))
#     page_module = menu[choice]
#     page_module.app(artifacts)

#     # --- Global Chatbot ---
#     st.sidebar.markdown("---")
#     st.sidebar.subheader("🤖 Assistant")

#     if "chat_history" not in st.session_state:
#         st.session_state["chat_history"] = []

#     user_query = st.sidebar.text_area("Ask me about gallstone risk:", height=80, key="chat_input")
#     if st.sidebar.button("Send", key="chat_send"):
#         if user_query.strip():
#             st.session_state["chat_history"].append(("You", user_query))
#             with st.spinner("Assistant is thinking..."):
#                 reply = query_local_llm(user_query)
#             st.session_state["chat_history"].append(("AI", reply))

#     for speaker, text in st.session_state["chat_history"][-8:]:
#         if speaker == "You":
#             st.sidebar.markdown(f"**You:** {text}")
#         else:
#             st.sidebar.markdown(f"**Assistant:** {text}")


# if __name__ == "__main__":
#     main()

