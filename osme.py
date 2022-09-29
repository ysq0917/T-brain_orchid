#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 08:45:20 2018
@author: n-kamiya
"""
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.applications.resnet import ResNet50, ResNet101
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Multiply, concatenate
from keras.layers import GlobalAveragePooling2D, Lambda
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.regularizers import l2
import matplotlib.image as mpimg
import numpy as np
import keras.backend as K
import pathlib
from PIL import Image
import cv2
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
# from tensorflow.keras.utils import plot_model
import pandas as pd

K.clear_session()
BATCH_SIZE = 16
train_nb = 1402
valid_nb = 350
test_nb = 438
num_classes = 219
img_size= 224

train_path = "train"
valid_path = "valid"

# create data generator 

train_datagen = ImageDataGenerator(rescale=1.0/255, 
                                   zoom_range=[1.0, 2.0],
                                   rotation_range=90,
                                   horizontal_flip=True,
                                   vertical_flip=True)
valid_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        seed=13)

valid_generator = valid_datagen.flow_from_directory(
        valid_path,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        seed=13)

# finetuning resnet50

input_tensor = Input(shape=(img_size, img_size, 3))
# base_model = VGG19(weights = "imagenet", include_top=False, input_tensor=input_tensor)
# base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=input_tensor)
base_model = ResNet101(weights="imagenet", include_top=False, input_tensor=input_tensor)

#for layer in base_model.layers:
#    layer.trainable = False
#    if isinstance(layer, keras.layers.normalization.BatchNormalization):
#        layer._per_input_updates = {}
        
# Implementation of OSME module
split = Lambda( lambda x: tf.split(x, num_or_size_splits=2, axis=3))(base_model.output)

def osme_block(in_block, ch, ratio=16):
    z = GlobalAveragePooling2D()(in_block) # 1
    x = Dense(ch//ratio, activation='relu')(z) # 2
    x = Dense(ch, activation='sigmoid')(x) # 3
    return Multiply()([in_block, x]) # 4

s_1 = osme_block(split[0], split[0].shape[3])
s_2 = osme_block(split[1], split[1].shape[3])
fc1 = Flatten()(s_1)
fc2 = Flatten()(s_2)
fc1 = Dense(1024, name='fc1')(fc1)
fc2 = Dense(1024, name='fc2')(fc2)
fc = concatenate([fc1, fc2]) # fc1 + fc2
prediction = Dense(num_classes, activation='softmax', name='prediction')(fc)
model = Model(inputs=base_model.input, outputs=prediction)

opt = SGD(learning_rate=0.001, momentum=0.9, decay=0.0005)
#model.load_weights("/home/n-kamiya/models/model_without_MAMC/model_osme_vgg_imagenet.best_loss.hdf5")  
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

# plot_model(model, to_file="model.png", show_shapes=True, dpi=300)

# implement checkpointer and reduce_lr (to prevent overfitting)
# checkpointer = ModelCheckpoint(filepath='model_osme_vgg19.best_loss.hdf5', verbose=1, save_best_only=True)
# checkpointer = ModelCheckpoint(filepath='model_osme_resnet50.best_loss.hdf5', verbose=1, save_best_only=True)
checkpointer = ModelCheckpoint(filepath='model_osme_resnet101.best_loss.hdf5', verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                  patience=5, min_lr=0.0000001)

# es_cb = EarlyStopping(patience=11)

# fit_generator
history = model.fit_generator(train_generator,
                    steps_per_epoch=train_nb//BATCH_SIZE,
                    epochs=60,
                    validation_data=valid_generator,
                    validation_steps=64,
                    verbose=1,
                    callbacks=[reduce_lr, checkpointer])