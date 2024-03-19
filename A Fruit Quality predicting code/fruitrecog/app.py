import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
#import fakeorginal as fo
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import rippen as frt
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
#import serial
import os
import cv2
from matplotlib import pyplot as plt


from time import sleep
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
import cv2
import numpy as np
from random import choice
from tensorflow.keras.models import load_model


REV_CLASS_MAP = {
    0: "Apple",
    1: "Banana",
    2: "Grape",
    3: "Orange",
    4: "Mango",
    5: "Strawberry",
    6: "apple1",
    7: "Banana1",
    8: "Pomegranate",
}

def mapper(value):
    return REV_CLASS_MAP[value]

def imagetest(fname):
    img_shape = (225, 225)
    model = load_model("fruits.h5")
    frame = cv2.imread(fname)#cap.read()
    

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_shape)

    pred = model.predict(np.array([img]))
    move_code = mapper(np.argmax(pred[0]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Fruit Detected : " + move_code, (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Fruit Recognition", img)
    k = cv2.waitKey(10)
    
    return(move_code)
def histogramImage(file_name,image_height,image_width,DPI):
    img = cv2.imread(file_name)
    base = os.path.basename(file_name)
    name = os.path.splitext(base)[0]
    ext = os.path.splitext(base)[1]
    histb = cv2.calcHist([img],[0],None,[256],[0,256])
    print(histb)
    histg = cv2.calcHist([img],[1],None,[256],[0,256])
    histr = cv2.calcHist([img],[2],None,[256],[0,256])
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    
    plt.xlabel('Pixel intensity')
    plt.ylabel('Number of pixels',horizontalalignment='left',position=(1,1))    
    figure = plt.gcf() # get current figure
    ax1.fill(histr,'r');
    ax2.fill(histg,'g');
    ax3.fill(histb,'b');
    figure.set_size_inches(image_width, image_height)#in inches
    # when saving, specify the DPI
    new_file_name = "histogram_"+name+ext
    print ("\nThe filename is: "+new_file_name)
    plt.savefig(new_file_name, dpi = DPI, bbox_inches='tight')
    plt.setp([a.get_xticklabels() for a in f.axes[:-5]], visible=True)
    plt.show()
    
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='fruits.h5'

# Load your trained model
model = load_model(MODEL_PATH)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    res=""
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = imagetest(file_path)
        cname=""
        print(file_path)
        frame = cv2.imread(file_path)#cap.read()
        frs=frt.fruit(frame)
        print(preds)
        histogramImage(file_path,4,8,600)
        
            
        return preds+" , Fruit Rippen Status "+frs
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
