import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load the pretrained model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

# Load and preprocess the image
image_path = "handwritten_sample.jpeg"  # Replace with your image file
image = Image.open(image_path).convert("RGB")
mage = image.resize((384, 384))

# Convert image to tensor
pixel_values = processor(image, return_tensors="pt").pixel_values

# Generate text output
with torch.no_grad():
    generated_ids = model.generate(
    pixel_values,
    max_length=50,  # Limit output length
    num_beams=5,  # Use beam search for better accuracy
    early_stopping=True  # Stop when the model finds a confident result
)

# Decode the text
extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Extracted Handwritten Text:\n", extracted_text)
# from PIL import Image
# import pytesseract

# # Set Tesseract path
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# # Load image
# image_path = "handwritten_sample1.jpeg"
# image = Image.open(image_path)

# # Extract text
# extracted_text = pytesseract.image_to_string(image)

# print("Extracted Handwritten Text:\n", extracted_text)


