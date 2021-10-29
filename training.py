#!/usr/bin/env python
# coding: utf-8

# In[10]:


from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import os.path


# In[11]:



weights='imagenet'
base_model = InceptionV3(weights=weights, include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


for layer in model.layers[:-2]:
    layer.trainable = False
    

    # compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



# In[12]:



train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(r"C:\Users\lalith kumar\Desktop\Rice leaf vjhack",
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical')
train_generator


# In[13]:


model.fit_generator(
        train_generator,
        epochs=10)


# In[ ]:


import cv2 as cv
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# In[19]:


import tensorflow as tf
import numpy as np
img_path = r'C:\Users\lalith kumar\Desktop\Rice leaf vjhack\IMG_2992'
img=cv.imread(img_path)
face=cv.resize(img,(244,244))
im=tf.keras.preprocessing.image.img_to_array(face)
img_array=np.array(im)
img_array=preprocess_input(img_array)
img_array=np.expand_dims(img_array,axis=0)
pred=[]
pred=model.predict(img_array)[0]
l=max(pred)
li=["Brown Spot","Hispa","Leaf blast"]
pi=[(0,255,0),(0,0,255),(0,0,225)]
for x in range(0,3):
 if(l==pred[x]):
     v=x
img=cv.resize(img,(700,500))        
cv.putText(img, li[v], (0,20), cv.FONT_HERSHEY_TRIPLEX, 1.0, pi[v], 2)
cv.imshow('pic', img)      
cv.waitKey(0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




