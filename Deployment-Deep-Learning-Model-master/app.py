from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.utils import load_img, img_to_array 
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = r'C:\Users\ST-0010\Music\Complete\Cotton Diesease\CODE\FRONT END\Deployment-Deep-Learning-Model-master\model_vgg16 (1).h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
# print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = tf.keras.utils.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds)
        # print(type(preds))
        resultx = None
        for j in preds:
            for i in range(len(j)):
                print(i)
                if j[i] > 0.5:
                    print(j[i])
                    if i == 0:
                        resultx = "diseased cotton leaf"
                    elif i == 1:
                        resultx = "diseased cotton plant"
                    elif i == 2:
                        resultx = "fresh cotton leaf"
                    elif i == 3:
                        resultx = "fresh cotton plant"
                    else:
                        resultx = "unsure of plant type"
                else:
                    pass
            # return result

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)           # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # print(pred_class)
        result = str(resultx)              # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

