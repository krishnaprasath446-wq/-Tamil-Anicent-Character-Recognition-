# Cell 1: Imports and FastAPI App Setup

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model

# Initialize the FastAPI app
app = FastAPI(title="Vatteluthu OCR API")

# Allow all origins for simplicity (you can restrict this in production)
# This lets our index.html file talk to the server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cell 2: Loading the Model and Translation Dictionaries

# --- Load Model and Mappings at startup ---
MODEL_PATH = 'vatteluthu_model.keras'
MAPPINGS_PATH = 'int_to_class.pkl'
IMAGE_SIZE = 64

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the class mappings (int -> folder_name)
try:
    with open(MAPPINGS_PATH, 'rb') as f:
        int_to_class = pickle.load(f)
    print("Class mappings loaded successfully.")
except Exception as e:
    print(f"Error loading class mappings: {e}")
    int_to_class = None

# --- NEW: Final Translation Dictionary (Folder Name -> Tamil Character) ---
# This dictionary translates the dataset's folder names to modern Tamil characters.
FOLDER_TO_TAMIL = {
    '1': 'ர', '2': 'ட', '3': 'ம', '4': 'ப', '5': 'ஒ', '6': 'ய', '7': 'ல', 
    '8': 'ன', '9': 'ற', '10': 'க', '11': 'в', '12': 'ண', '13': 'த', '14': 'ச',
    '15': 'ங', '16': 'ள', '17': 'தி', '18': 'வி', '19': 'றி', '20': 'லி', 
    '21': 'னி', '22': 'யி', '23': 'ழி', '24': 'பி', '25': 'ரி', '26': 'சி',
    '27': 'ணி', '28': 'மி', '29': 'கி', '30': 'டு', '31': 'கு', '32': 'ளு',
    '33': 'லு', '34': 'மு', '35': 'ணு', '36': 'னூ', '37': 'ஞ', '38': 'பு',
    '39': 'று', '40': 'னு', '41': 'சு', '42': 'து', '43': 'வு', '44': 'யு',
    '45': 'ழு', '46': 'டி', '47': 'ளி', '48': 'எ', '49': 'ழ', '50': 'கீ',
    '51': 'றூ', '52': 'மீ', '53': 'வீ', '54': 'நூ', '55': 'றீ', '56': 'தீ',
    '57': 'கூ', '58': 'தூ', '59': 'சூ', '60': 'யீ', '61': 'லூ', '62': 'உ',
    '63': 'அ', '64': 'ழீ', '65': 'யூ', '66': 'சீ', '67': 'ணீ', '68': 'ஆ',
    '69': 'ளு', '70': 'இ', '71': 'ழூ'
    # NOTE: You will need to add the rest of the mappings up to 208
    # for the model to recognize all possible characters.
}

# Cell 3: The Image Preprocessing Function

# --- Preprocessing function (must be identical to the one in training) ---
def preprocess_for_prediction(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        resized_img = cv2.resize(binary_img, (IMAGE_SIZE, IMAGE_SIZE))
        normalized_img = resized_img / 255.0
        reshaped_img = normalized_img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
        return reshaped_img
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

# Cell 4: The Prediction API Endpoint

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Vatteluthu Character Recognition API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model or not int_to_class:
        return {"error": "Model or mappings not loaded. Check server logs."}
        
    contents = await file.read()
    processed_image = preprocess_for_prediction(contents)
    
    if processed_image is None:
        return {"error": "Could not process the uploaded image."}

    # Make prediction
    prediction_probs = model.predict(processed_image)
    prediction_index = np.argmax(prediction_probs)
    
    # --- UPDATED: Two-Step Lookup ---
    
    # 1. Get the folder name from the model's prediction index
    predicted_folder_name = int_to_class.get(prediction_index, "Unknown")
    
    # 2. Get the final Tamil character from the folder name using our new dictionary
    final_character = FOLDER_TO_TAMIL.get(predicted_folder_name, "?")
    
    confidence = float(np.max(prediction_probs))

    return {
        "predicted_character": final_character,
        "confidence": f"{confidence:.2f}"
    }