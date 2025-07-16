from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import traceback

def debug_solve_local(image_path: str) -> str:
    print("--- Starting Local Solver ---")
    try:
        print(f"1. Attempting to open image at: '{image_path}'")
        image = Image.open(image_path).convert("RGBA")
        print("   - Image opened successfully.")

        print("2. Loading TrOCR processor from Hugging Face...")
        processor = TrOCRProcessor.from_pretrained("anuashok/ocr-captcha-v3", use_fast=True)
        print("   - Processor loaded.")

        print("3. Loading VisionEncoderDecoder model from Hugging Face...")
        model = VisionEncoderDecoderModel.from_pretrained("anuashok/ocr-captcha-v3")
        print("   - Model loaded.")

        print("4. Preparing image for the model...")
        background = Image.new("RGBA", image.size, (255, 255, 255))
        combined = Image.alpha_composite(background, image).convert("RGB")
        pixel_values = processor(combined, return_tensors="pt").pixel_values
        print("   - Image prepared.")

        print("5. Running model inference to generate text...")
        generated_ids = model.generate(pixel_values)
        print("   - Inference complete.")

        print("6. Decoding the result...")
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("   - Decoding complete.")
        
        return generated_text

    except FileNotFoundError:
        print(f"ERROR: The file was not found at '{image_path}'")
        print("Please make sure the file exists and the path is correct.")
        return ""
    except Exception as e:
        print(f"AN UNEXPECTED ERROR OCCURRED: {e} !!!")
        traceback.print_exc()
        return ""
