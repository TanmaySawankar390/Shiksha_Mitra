import os
import io
from flask import Flask, request, jsonify
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Configure the API
load_dotenv()

# Fetch API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")

# Check if API key is found
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# Configure the API
genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the API!"})

def extract_qa_from_image(image):
    """
    Extracts multiple questions and answers from an image.

    Args:
        image (PIL.Image): Image object.

    Returns:
        list: A list of dictionaries with 'question' and 'answer'.
    """
    # Prompt to ensure structured output
    prompt = (
        "Extract all question-answer pairs from the image. "
        "Return the output in this structured format:\n"
        "Q1: <question>\nA1: <answer>\n"
        "Q2: <question>\nA2: <answer>\n"
        "Continue this format for all questions."
    )

    response = model.generate_content([prompt, image])

    if response and response.text:
        extracted_text = response.text.strip()
        qa_list = []
        lines = extracted_text.split("\n")

        for i in range(0, len(lines) - 1, 2):  # Process two lines at a time (Q and A)
            if lines[i].startswith("Q") and lines[i + 1].startswith("A"):
                question = lines[i].split(":", 1)[1].strip()
                answer = lines[i + 1].split(":", 1)[1].strip()
                qa_list.append({"question": question, "answer": answer})

        return qa_list if qa_list else [{"question": "No questions detected", "answer": "No answers detected"}]

    return [{"question": "No questions detected", "answer": "No answers detected"}]

@app.route('/extract_qa', methods=['POST'])
def extract_qa():
    """
    API endpoint to extract Q&A pairs from an image.

    Supports:
    - Image upload via 'image' field (multipart/form-data)
    - Image path via JSON { "image_path": "path/to/image.jpg" }
    """
    # Handle image upload
    if 'image' in request.files:
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))
        qa_list = extract_qa_from_image(image)
        return jsonify({"questions_answers": qa_list}), 200

    # Handle image path
    if request.is_json:
        data = request.get_json()
        if "image_path" in data:
            image_path = data["image_path"]
            if not os.path.exists(image_path):
                return jsonify({"error": "Image file not found"}), 400
            image = Image.open(image_path)
            qa_list = extract_qa_from_image(image)
            return jsonify({"questions_answers": qa_list}), 200

    return jsonify({"error": "No valid image provided"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8080)
