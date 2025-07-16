import streamlit as st
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os

# --- Model Loading ---
# This function will only run ONCE, and the result will be cached.
@st.cache_resource
def load_model():
    print("--- LOADING MODELS (this should only happen once) ---")
    processor = TrOCRProcessor.from_pretrained("anuashok/ocr-captcha-v3", use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained("anuashok/ocr-captcha-v3")
    print("--- MODELS LOADED ---")
    return processor, model

# --- Inference Function ---
# This function now ACCEPTS the models as arguments instead of loading them.
def solve_captcha(image_path, processor, model):
    try:
        image = Image.open(image_path).convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255))
        combined = Image.alpha_composite(background, image).convert("RGB")
        
        pixel_values = processor(combined, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text
    except Exception as e:
        st.error(f"An error occurred during inference: {e}")
        return ""

# --- Streamlit UI ---
st.title("Captcha Solver")

# Load the models using the cached function
processor, model = load_model()

uploaded_file = st.file_uploader("Choose a captcha image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Save, display, and process the file
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption='Uploaded Captcha')
    
    with st.spinner('Solving...'):
        result = solve_captcha(temp_file_path, processor, model)
    
    st.success(f'**Result:** {result}')

    os.remove(temp_file_path)
