from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
# Keras

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#GPU priority set
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH, IMG_SIZE = 'models/LeNet_BestModel.h5', 32
MODEL_PATH, IMG_SIZE = 'models/AlexNet_BestModel.h5', 224
#MODEL_PATH, IMG_SIZE = 'models/Gnet_BestModel.h5', 32

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()
print('Model loaded. Start serving...')
stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)
print(short_model_summary)
# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)  # convert to array
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    x = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    #img = image.load_img(img_path, target_size=(32, 32))

    # Preprocessing the image
    #x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    preds = np.argmax(preds,axis=1)
    #preds = model.predict_classes(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        #pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = pred_class
        # Convert to string
        if preds == 0:
            return "Coffee Rust"
        elif preds == 1:
            return "Sooty Mold"
        elif preds == 2:
            return "Healthy Leaf"


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
