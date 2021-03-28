
from flask import Flask, render_template, url_for, request, jsonify, make_response,flash,redirect, session
#import pyrebase
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as im
import matplotlib.pyplot as plt
import numpy as np
import os

import scipy.io as scio
import numpy as np    
import os
import matplotlib.pyplot as plt
import math
import re
#from scipy.misc import imsave
from cv2 import cv2
#from scipy import ndimage, misc # not required
from numpy import unravel_index
from operator import sub
import keras
import tensorflow as tf
from keras.layers import Reshape
from keras import backend as K
from keras import regularizers, optimizers
from keras.models import Model
import keras
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Lambda 
from keras.utils import to_categorical
import skimage.restoration as sr

UPLOAD_FOLDER = 'static/assets/uploads/'

class_labels = ['DME','NORMAL']
#allow only the selected extension file.
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# load model
classification_model = load_model("./final_model.h5")


##############################
data_shape = 216*64
weight_decay = 0.0001
# Defines the input tensor
inputs = Input(shape=(216,64,1))

L1 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(inputs)
L2 = BatchNormalization()(L1)
L2 = Activation('relu')(L2)
#L3 = Lambda(maxpool_1,output_shape = shape)(L2)
L3 = MaxPooling2D(pool_size=(2,2))(L2)
L4 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L3)
L5 = BatchNormalization()(L4)
L5 = Activation('relu')(L5)
#L6 = Lambda(maxpool_2,output_shape = shape)(L5)
L6 = MaxPooling2D(pool_size=(2,2))(L5)
L7 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L6)
L8 = BatchNormalization()(L7)
L8 = Activation('relu')(L8)
#L9 = Lambda(maxpool_3,output_shape = shape)(L8)
L9 = MaxPooling2D(pool_size=(2,2))(L8)
L10 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L9)
L11 = BatchNormalization()(L10)
L11 = Activation('relu')(L11)
L12 = UpSampling2D(size = (2,2))(L11)
#L12 = Lambda(unpool_3,output_shape = unpool_shape)(L11)
L13 = Concatenate(axis = 3)([L8,L12])
L14 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L13)
L15 = BatchNormalization()(L14)
L15 = Activation('relu')(L15)
L16 = UpSampling2D(size= (2,2))(L15)
#L16 = Lambda(unpool_2,output_shape=unpool_shape)(L15)
L17 = Concatenate(axis = 3)([L16,L5])
L18 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L17)
L19 = BatchNormalization()(L18)
L19 = Activation('relu')(L19)
#L20 = Lambda(unpool_1,output_shape=unpool_shape)(L19)
L20 = UpSampling2D(size=(2,2),name = "Layer19")(L19)
L21 = Concatenate(axis=3)([L20,L2])
L22 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L21)
L23 = BatchNormalization()(L22)
L23 = Activation('relu')(L23)
L24 = Conv2D(8,kernel_size=(1,1),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L23)
L = Reshape((data_shape,8),input_shape = (216,64,8))(L24)
L = Activation('softmax')(L)
model = Model(inputs = inputs, outputs = L)

###################################
max_rows = 216
hval = 10
alpha = 15
beta = 1


model.load_weights("Relaynet_sample_weights_denoised_lr_e2_testing_bs_20.hdf5")
def reshape(ndarray):
    ndarray_reshaped = np.array(ndarray.tolist())
    ndarray_reshaped = ndarray_reshaped.reshape(ndarray.shape).transpose(2,0,1)
    print (ndarray_reshaped.shape)
    return ndarray_reshaped

def crop_image(image,left_bound,right_bound):
    image = image[:,left_bound:right_bound]
    return image

def resize_image(image,min_value,max_value,diff_value):
    extra = max_rows - diff_value
    min_value = int(min_value - math.ceil(extra/2.0))
    max_value = int(max_value + math.floor(extra/2.0))
    image = image[min_value:max_value,:]
    return image
def denoiseImage(image):
    maxvalue = np.max(image)
    newimage = image*(255.0/maxvalue).astype(np.uint8)
    denoised = sr.denoise_nl_means(newimage, multichannel=False, h=hval)
    denoised = denoised - (alpha*beta)
    denoised[denoised<0]=0
    denoised = denoised.astype(np.uint8)
    return denoised
    
# it should return image which later I woul append in the variable



app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def login():
    return render_template('login.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/final_report', methods=['GET', 'POST'])

def data_():
    if request.method == 'POST':
        
        name = request.form.get('name')
        age = request.form.get('age')
        file = request.form.get('file')

        
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        

        filename = 'static/assets/uploads/'+filename
        print(filename)
        

        image = cv2.imread(filename,0)
        max_rows = 216
        #cropped_img = crop_image(image,130,630)
        res_img = resize_image(image,150,250,100)
        #res_img = resize_image(image,200,300,100)
        res_crop_img =[]
        for x in range(7):
            temp = crop_image(res_img,x*64,(x+1)*64)
            res_crop_img.append(temp)
        denoisedimages = []
        hval = 10
        alpha = 15
        beta = 1
        for image in res_crop_img:
            denoisedimages.append(denoiseImage(image))
        denoisedimages=np.array(denoisedimages)
        print(denoisedimages.shape[0])
        denoisedimages = denoisedimages.reshape(denoisedimages.shape[0],216,64,1)
        testing_images = []
        for x in denoisedimages:
            x = np.squeeze(x,axis=2)
            testing_images.append(x.reshape((1,216,64,1)))
        predictions = []
        for y in testing_images:
            y = model.predict(y)
            y = np.squeeze(y,axis=0)
            predictions.append(np.reshape(y,(216,64,8)))
        print("pred",predictions)
        output = []
        for z in range(7):
            output.append(np.zeros((216,64)))
        
        for h in range(7):
            for i in range(216):
                for j in range(64):
                    index = np.argmax(predictions[h][i][j])
                    output[h][i][j] = index
        print("OUT:", output)
        color= []
        for a in range(7):
            color.append(np.zeros((216,64,3)))
        c0 = 0
        c1 = 0
        c2 = 0
        c3 = 0
        c4 = 0
        c5 = 0
        c6 = 0
        c7 = 0

        for i in range(7):
    
            for j in range(216):
                for k in range(64):
                    if(output[i][j][k]==0):
                        c0 = c0 + 1
                        color[i][j][k] = [0,0,0]
                    if(output[i][j][k]==1):
                        c1 = c1 + 1
                        color[i][j][k] = [128,0,0]
                    if(output[i][j][k]==2):
                        c2 = c2 + 1
                        color[i][j][k] = [0,128,0]
                    if(output[i][j][k]==3):
                        c3 = c3 + 1
                        color[i][j][k] = [128,128,0] 
                    if(output[i][j][k]==4):
                        c4 = c4 + 1
                        color[i][j][k] = [0,128,128]
                    if(output[i][j][k]==5):
                        c5 = c5 + 1
                        color[i][j][k] = [64,220,0]
                    if(output[i][j][k]==6):
                        c6 = c6 + 1
                        color[i][j][k] = [192,0,0]
                    if(output[i][j][k]==7):
                        c7 = c7 + 1
                        color[i][j][k] = [64,128,0]
            c0 = 0
            c1 = 0
            c2 = 0
            c3 = 0
            c4 = 0
            c5 = 0
            c6 = 0
            c7 = 0
            
        print("Color:", color)
        im_h = cv2.hconcat(color)
        im_h = np.ceil(im_h/255)
        plt.imsave("./static/assets/uploads/"+name+".png",im_h)

        img = im.load_img(filename, target_size=(160, 160))
        img_tensor = im.img_to_array(img)                    # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
        pred = classification_model.predict(img_tensor)
        predicted_class = np.argmax(pred)
        filename2 = "./static/assets/uploads/"+name+".png"

        data={'name':name,'age':age,'filename':filename}
        result = class_labels[predicted_class]
        return render_template('final_report.html', data=data, name=name,filename2=filename,filename1 = filename2, age=age, result=result)




app.run(debug=False)