#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import os
import requests
import zipfile
import shutil


# In[2]:


DATA_DIR = "/data/krf/dataset/ChinaSet_AllFiles/CXR_png"
MASK_DIR = "/data/krf/dataset/mask"
ROOT_DIR = "/data/krf/model/rsna"


# In[27]:


<<<<<<< HEAD
IMG_SIZE =256
=======
IMG_SIZE = 256
>>>>>>> 4c868fc6388ae085ba408226a0a66322a948e61c
def loadDataset():
    image_fp = os.listdir(DATA_DIR)
    image_fp.sort()
    images = []
    for img_nm in image_fp:
        tmp = cv2.imread(os.path.join(DATA_DIR,img_nm))
        b,g,r = cv2.split(tmp)
        res = cv2.resize(b,(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.uint8)
        images.append(res[:,:,np.newaxis])
    images = np.asarray(images)
    print(images.shape)
    mask_fp = os.listdir(MASK_DIR)
    mask_fp.sort()
    masks = []
    for msk_nm in mask_fp:
        tmp = cv2.imread(os.path.join(MASK_DIR,msk_nm))
        b,g,r = cv2.split(tmp)
        res = cv2.resize(b,(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.bool)
        masks.append(res[:,:,np.newaxis])
    masks = np.asarray(masks)
    print(masks.shape)
    return images,masks


# In[28]:

images,masks = loadDataset()
#get_ipython().run_cell_magic('time', '', 'images,masks = loadDataset()')


# In[5]:


#delete some image without mask file
# image_fp = os.listdir(DATA_DIR)
# image_fp.sort()
# mask_fp = os.listdir(MASK_DIR)
# mask_fp.sort()
# j = 0
# for i in range(len(image_fp)):
#     #print(mask_fp[j][:-9],image_fp[i][:-4])
#     if mask_fp[j][:-9] != image_fp[i][:-4]:
#         print(image_fp[i])
#         os.remove(os.path.join(DATA_DIR,image_fp[i]))
#         i += 1
#         j -= 1
#     j+=1


# In[29]:


img_index = 3
plt.figure(figsize=(30, 20))
plt.subplot(1,3,1)
plt.imshow(images[img_index,:,:,0], cmap=plt.cm.bone)
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(masks[img_index,:,:,0], cmap=plt.cm.bone)
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(cv2.bitwise_and(images[img_index,:,:,0], images[img_index,:,:,0], 
                           mask=masks[img_index,:,:,0].astype(np.uint8)), cmap=plt.cm.bone)
plt.axis('off')


# In[7]:


# download MD.ai's dilated unet implementation 
# UNET_URL = 'https://s3.amazonaws.com/md.ai-ml-lessons/unet.zip'
# UNET_ZIPPED = 'unet.zip'

# if not os.path.exists(UNET_ZIPPED): 
#     r = requests.get(UNET_URL, stream=True)
#     if r.status_code == requests.codes.ok:
#         with open(UNET_ZIPPED, "wb") as f:
#             shutil.copyfileobj(r.raw, f)
#     else:
#         r.raise_for_status()

# with zipfile.ZipFile(UNET_ZIPPED) as zf:
#     zf.extractall()


# In[30]:


from unet import dataset
from unet import dilated_unet
from unet import train


# In[31]:


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
<<<<<<< HEAD
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

CONFIG_FP = 'unet/configs/krf.json'
=======
config.gpu_options.per_process_gpu_memory_fraction = 0.15
set_session(tf.Session(config=config))

CONFIG_FP = 'unet/configs/11.json'
>>>>>>> 4c868fc6388ae085ba408226a0a66322a948e61c
name = os.path.basename(CONFIG_FP).split('.')[0]
print(name)


# In[32]:


import json
with open(CONFIG_FP,'r') as f:
    config = json.load(f)


# In[ ]:


history = train.train(config,name,images,masks,num_epochs = 50)


# In[12]:


import matplotlib.pyplot as plt

print(history.history.keys())

plt.figure()
plt.plot(history.history['acc'], 'orange', label='Training accuracy')
plt.plot(history.history['val_acc'], 'blue', label='Validation accuracy')
plt.plot(history.history['loss'], 'red', label='Training loss')
plt.plot(history.history['val_loss'], 'green', label='Validation loss')
plt.legend()
plt.show()
plt.savefig(ROOT_DIR+"/history.png")

# In[13]:


from keras.models import load_model
import keras.backend as K

model_name = 'unet/trained/model_'+name+'.hdf5'
print(model_name)
model = load_model(model_name, custom_objects={'dice': train.dice, 'iou': train.iou})


# In[15]:


#images, masks = dataset.load_images(imgs_anns_dict)
import random
plt.figure(figsize=(20, 10))

img_index = random.choice(range(len(images)))

plt.subplot(1,4,1)
random_img = images[img_index,:,:,0]
plt.imshow(random_img, cmap=plt.cm.bone)
plt.axis('off')
plt.title('Lung X-Ray')

plt.subplot(1,4,2)
random_mask = masks[img_index,:,:,0]
plt.imshow(random_mask, cmap=plt.cm.bone)
plt.axis('off')
plt.title('Mask Ground Truth')

random_img_2 = np.expand_dims(np.expand_dims(random_img, axis=0), axis=3)
mask = model.predict(random_img_2)[0][:,:,0] > 0.5
plt.subplot(1,4,3)
plt.imshow(mask, cmap=plt.cm.bone)
plt.axis('off')
plt.title('Predicted Mask')

plt.subplot(1,4,4)
plt.imshow(cv2.bitwise_and(random_img, random_img, mask=mask.astype(np.uint8)), cmap=plt.cm.bone)
plt.axis('off')
plt.title('Predicted Lung Segmentation')

plt.savefig(ROOT_DIR+"/seg_test.png")
