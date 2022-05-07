import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse 
import tensorflow as tf
import re
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input

def decode (predicted_arr):
  train_path = './20valid'

  all_birds_cat = np.array(list(sorted(os.listdir(train_path))))
  for i, pred in enumerate(predicted_arr):
    return all_birds_cat[np.argmax(pred)]

app = FastAPI()
@app.get('/') #basic get view
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}
  
@app.get('/predict', response_class=HTMLResponse)
def take_inp():
  loaded_model = tf.keras.models.load_model('mobile_net.h5') #load the saved model 
  predict_image_path = './20valid/AMERICAN REDSTART/1.jpg'

  img = load_img(predict_image_path, target_size=(224,224))
  img = img_to_array(img)
  img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
  img = preprocess_input(img)

  predictions = loaded_model.predict(img)
  label = decode(predictions)

  return f"The predicted label is {label}"