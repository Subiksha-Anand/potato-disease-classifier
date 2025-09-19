from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import tf_keras as k3
from tensorflow.keras.models import load_model
import requests

app= FastAPI()
endpoint = "http://localhost:8504/v1/models/potato_model:predict"

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
    
    json_data = {
        "instances":img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)

    print("Response from TensorFlow Serving:", response.json())  # Debugging line

    if response.status_code != 200:
        return {"error": "Failed to get prediction from TensorFlow Serving", "details": response.json()}

    if "predictions" not in response.json():
        return {"error": "Unexpected response format", "details": response.json()}

    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": confidence
    }
   

if __name__ =="__main__":
    uvicorn.run(app,host='localhost',port=5002)