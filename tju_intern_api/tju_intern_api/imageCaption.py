#!/usr/bin/env python
# coding: utf-8

# # IMAGE CAPTIONING
# 

# In[1]:


import pandas as pd
import numpy as np
import json
import pickle
import tensorflow.keras 
from time import time
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras .models import Model,load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input,Dense,Dropout,Embedding,LSTM
from tensorflow.keras.layers import add
import re
import string
import matplotlib.pyplot as plt
import os
import sys


# In[2]:

cwd = os.getcwd()

descripitions = None
with open((os.path.join( cwd, 'tju_intern_api\descripitions_1.text')),"r")as f:
    descripitions = f.read()
    json_accept_str = descripitions.replace("'","\"")
    descripitions = json.loads(json_accept_str)


# In[3]:


total_words = []
for key in descripitions.keys():
    [total_words.append(i)for des in descripitions[key] for i in des.split()]


# In[5]:


import collections
counter = collections.Counter(total_words)
frq_cnt = dict(counter)
sort_frg_cnt = sorted(frq_cnt.items(),reverse= True,key= lambda x :x[1])


# In[6]:


threshold =10
sort_frg_cnt = [x for x in sort_frg_cnt if x[1] > threshold]
total_words = [x[0]for x in sort_frg_cnt]


# In[7]:


# ## image feature extraction

# In[8]:


model = ResNet50(weights="imagenet",input_shape = (224,224,3))
# model.summary()


# In[9]:


model_new = Model(model.input,model.layers[-2].output)


# In[10]:


def preproccess_img(img):
    img = image.load_img(img,target_size = (224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    return img


# In[11]:


def encode_img (img):
    img = preproccess_img(img)
    feature_vec  = model_new.predict(img).reshape((-1,))
    return feature_vec


# In[12]:


word_to_idx ={}
idx_to_word = {}


for i,word in enumerate(total_words):
    word_to_idx[word] = i+1
    idx_to_word[i+1] = word


# In[13]:


#two special word start seqn & end seqn
idx_to_word[1846] = 'startseq'
word_to_idx['startseq'] = 1846
idx_to_word[1847] = 'endseq'
word_to_idx['endseq'] = 1847


# In[14]:

model_path = (os.path.join( cwd, 'tju_intern_api\model\model40epochs.h5'))
model = load_model(model_path)


# # predictions

# In[15]:


def predict_caption(photo):
    in_text = 'startseq'
    # max_len
    for i in range(35):
        sequence = [word_to_idx[w]for w in in_text.split()if w in word_to_idx]
        # max_len = 35
        sequence = pad_sequences([sequence],maxlen= 35,padding="post")
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text += ' '+ word
        if word == 'endseq':
            break
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

def img2sent(testimg_path):
    encoding_testimg= encode_img(testimg_path).reshape((1,2048))
#     print(encoding_testimg)
    caption = predict_caption(encoding_testimg)
    return(caption)

if __name__ == '__main__':
    testimg_path = (os.path.join( cwd, 'dog.jpg'))
    print(img2sent(testimg_path))



