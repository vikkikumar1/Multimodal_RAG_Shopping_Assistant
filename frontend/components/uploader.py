import streamlit as st

def upload_section():
    return st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])