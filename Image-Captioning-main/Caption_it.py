#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import json

import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add


# In[2]:


model = load_model("model_weights/model_9.h5")
#model._make_predict_function()

# In[3]:


model_temp = ResNet50(weights="imagenet",input_shape=(224,224,3))


# In[4]:


#model_temp.summary()


# In[5]:


#Create a new model, by removing the last layer(output layer of 1000 classes) from the ResNet50
model_resnet = Model(model_temp.input,model_temp.layers[-2].output)
#model_resnet._make_predict_function()

# In[6]:


def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img) # 3-d
    img = np.expand_dims(img,axis=0) # 1 dimension is increased for batch_size
    # Normalisation
    img = preprocess_input(img) # preprocess_input imported from keras 
    return img


# In[7]:


def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1,feature_vector.shape[1]) # we have rehaped here as (1,2048)
    return feature_vector
    


# In[8]:


with open("storage/word_to_idx.pkl","rb") as w2i:
    word_to_idx = pickle.load(w2i)


# In[9]:


with open("storage/idx_to_word.pkl","rb") as i2w:
    idx_to_word = pickle.load(i2w)


# In[10]:


def predict_caption(photo):
    max_len = 35
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx] #convert word to number
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post') #padding i.e. empty space is filled with 0
        
        ypred = model.predict([photo,sequence]) # containing photo and sequence
        ypred = ypred.argmax() #Word with max prob always - Greedy Sampling
        word = idx_to_word[ypred] # covert number to word
        in_text += (' ' + word)
        
        if word == "endseq":
            break
    
    final_caption = in_text.split()[1:-1] # ignore first word and last word
    final_caption = ' '.join(final_caption) # join every word
    return final_caption


# In[11]:

def caption_this_image(image): #pass image path
    enc = encode_image(image)
    caption = predict_caption(enc)
    return caption #return caption



# In[12]:





# In[ ]:




