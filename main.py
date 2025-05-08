from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os
import requests

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# S3 model download URL
MODEL_URL = "https://kitish-whatsapp-bot-media.s3.ap-south-1.amazonaws.com/documentMessage_1746721456066.bin"  # Replace with actual URL
MODEL_PATH = "covid.h5"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from S3...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded.")

# Load the trained model
model = load_model(MODEL_PATH)
print("Model loaded.")

# Image preprocessing function
def preprocess(img_data):
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    img = img.resize((224, 224))  # Match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = preprocess(contents)

    prediction = model.predict(img_array)[0][0]  # Extract scalar from array
    predicted_class = int(prediction >= 0.5)     # Apply threshold
    confidence = float(prediction)               # Raw sigmoid output

    return {
        "class": predicted_class,
        "confidence": round(confidence, 4)
    }
