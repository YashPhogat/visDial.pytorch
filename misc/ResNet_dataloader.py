import numpy as np
import os
from urllib.request import urlopen,urlretrieve
from PIL import Image

from keras.models import load_model
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
# from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import time
import json
import h5py

def get_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
#
# file_path = '../script/data/visdial_params_demo.json'
#
# f = json.load(open(file_path, 'r'))
# itow = f['itow']
# img_info = f['img_train']
#
# img_list = []
# for i in img_info:
#     img_list.append(i['path'])
#
# data = np.zeros((20,7,7,512))
# f = h5py.File('data_demo.hdf5','w')
# f.create_dataset('data', data=data[0:1,:,:,:] , chunks=True, maxshape=(None,7,7,512))

img_height = 224
img_width = 224
base_model = VGG16(weights= 'imagenet', include_top=False, input_shape= (img_height,img_width,3))
a = np.zeros((2,224,224,3))
path = '../images/410561.jpg'
x = get_image(path)
print(x.shape)
path = '../images/410561.jpg'
x2 = get_image(path)
a[0:1,:,:,:] = x
a[1:2,:,:,:] = x2
preds = base_model.predict(a)
print(preds.shape)
#
# print('Predicted:', decode_predictions(preds, top=3)[0])
# t = time.time()
#
#
# idx = 0
# for i in range(len(img_list)):
#     path = '../images/dog1.jpg'
#     x = get_image(path)
#     data[idx,:,:,:] = base_model.predict(x)[0,:,:,:]
#     idx += 1
#     if idx%20 == 0:
#         f = h5py.File('data_demo.hdf5', 'a')
#         f['data'].resize(f['data'].shape[0] + data.shape[0], axis=0)
#         f['data'][-data.shape[0]:] = data
#         idx = 0
#
# t = time.time()-t
# print(t/1000)
#
