#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from PIL import Image

# Load logo
logo = Image.open("Gemini_Generated_Image_gi8m9lgi8m9lgi8m.png")

# Page config
st.set_page_config(page_title="LithoSense", page_icon=logo, layout="wide")

# Sidebar / Topbar
st.image(logo, width=100)
st.title("LithoSense")
st.subheader("AI-powered Gallstone Risk Prediction and Explainability Tool")

st.markdown("""
Welcome to **LithoSense**, a clinical decision-support tool that uses 
machine learning to predict gallstone risk, explain predictions with SHAP, 
and ensure fairness across patient subgroups.

👉 Navigate to the pages from the left sidebar.
""")


# In[3]:


# Run this in a Jupyter cell (will stream logs into the cell)
get_ipython().system('streamlit run app.ipynb')


# In[ ]:




