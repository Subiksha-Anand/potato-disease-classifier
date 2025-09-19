from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras as k3
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware


app= FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow only frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


MODEL = tf.keras.models.load_model(r"C:\Users\subik\OneDrive\Document\PROJECT\potato\saved_model\converted_model.keras")
CLASS_NAMES = ["Early Blight", "Late Blight","Healthy"]


@app.get("/ping")
async def ping():
    return "hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)  # Add batch dimension

    predictions = MODEL(img_batch)  # Directly call the model

    if isinstance(predictions, dict):  # If it's a dict, extract the first tensor
        predictions = list(predictions.values())[0]

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        "class": predicted_class,
        "confidence": float(confidence),
    }


if __name__ =="__main__":
    uvicorn.run(app,host='localhost',port=5002)

