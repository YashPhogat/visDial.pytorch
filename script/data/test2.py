import numpy as np
import time
import json
import h5py

f = h5py.File('vdl_img_vgg_demo.h5','r')
imgs = f['images_train']
print(np.mean(imgs[:,:,:,:]))