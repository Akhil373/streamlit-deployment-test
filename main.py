# app.py
import streamlit as st
from modelcode import debug_solve_local # Re-use your model code
import os

st.title("Captcha Solver")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a captcha image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the image
    st.image(uploaded_file, caption='Uploaded Captcha', use_column_width=True)
    st.write("")
    
    # Run inference
    with st.spinner('Solving...'):
        result = debug_solve_local(temp_file_path)
    
    st.success(f'**Result:** {result}')

    # Clean up the temp file
    os.remove(temp_file_path)
