import json
import numpy as np
from fastapi import FastAPI, File, UploadFile
from starlette.responses import HTMLResponse 
import tensorflow as tf
import re
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO

def decode (predicted_arr):
  train_path = './20valid'

  all_birds_cat = np.array(list(sorted(os.listdir(train_path))))
  for i, pred in enumerate(predicted_arr):
    confidence = np.round(np.max(pred) * 100, 2)
    return (all_birds_cat[np.argmax(pred)], confidence)

def read_imagefile(file) -> Image.Image:
  image = Image.open(BytesIO(file))
  return image

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/') #basic get view
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}
  
@app.post('/predict', response_class=HTMLResponse)
async def take_inp(file: UploadFile = File(...)):
  loaded_model = tf.keras.models.load_model('./mobile_net.h5') #load the saved model
  
  img = read_imagefile(await file.read())
  img = np.asarray(img.resize((224, 224)))[..., :3]
  img = np.expand_dims(img, 0)
  img = img / 127.5 - 1.0

  predictions = loaded_model.predict(img)
  label, confidence = decode(predictions)

  return json.dumps({"label": label, "confidence": confidence}, indent=4)