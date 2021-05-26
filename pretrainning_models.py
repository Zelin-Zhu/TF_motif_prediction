# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 20:34:56 2021

@author: ASUS
"""
import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow.keras import regularizers

#设置参数
detctor_length=5
num_detector=16
num_hidden_unit=16
weight_decay = 0.01

##2.建立model1
model1=tf.keras.Sequential()
#model.add(layers.Conv1D(num_detector,detctor_length,input_shape=(train_x.shape[1:]),activation='relu'))
model1.add(layers.Conv1D(num_detector,detctor_length,input_shape=((100,4)),activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
model1.add(layers.GlobalMaxPool1D())
#model.add(layers.Dense(num_hidden_unit,activation='relu'))
model1.add(layers.Dense(num_hidden_unit,activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
model1.add(layers.Dropout(0.5))
model1.add(layers.Dense(1,activation='sigmoid'))
##3. 模型编译
model1.compile(optimizer=tf.keras.optimizers.RMSprop(),
             loss='binary_crossentropy',
             metrics=['acc'])
path="D:/workspace of spyder/毕业设计/my project data/model_file/pretrainning_models/"
model1.save(path+'model1.h5')

##建立model2
detctor_length=5
num_detector=32
num_hidden_unit=32
weight_decay = 0.01

model2=tf.keras.Sequential()
#model.add(layers.Conv1D(num_detector,detctor_length,input_shape=(train_x.shape[1:]),activation='relu'))
model2.add(layers.Conv1D(num_detector,detctor_length,input_shape=((100,4)),activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
model2.add(layers.GlobalMaxPool1D())
#model.add(layers.Dense(num_hidden_unit,activation='relu'))
model2.add(layers.Dense(num_hidden_unit,activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(1,activation='sigmoid'))
## 模型编译
model2.compile(optimizer=tf.keras.optimizers.RMSprop(),
             loss='binary_crossentropy',
             metrics=['acc'])
path="D:/workspace of spyder/毕业设计/my project data/model_file/pretrainning_models/"
model2.save(path+'model2.h5')


##建立model3
detctor_length=16
num_detector=32
num_hidden_unit=32
weight_decay = 0.01

model3=tf.keras.Sequential()
#model.add(layers.Conv1D(num_detector,detctor_length,input_shape=(train_x.shape[1:]),activation='relu'))
model3.add(layers.Conv1D(num_detector,detctor_length,input_shape=((100,4)),activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
model3.add(layers.GlobalMaxPool1D())
#model.add(layers.Dense(num_hidden_unit,activation='relu'))
model3.add(layers.Dense(num_hidden_unit,activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
model3.add(layers.Dropout(0.5))
model3.add(layers.Dense(1,activation='sigmoid'))
## 模型编译
model3.compile(optimizer=tf.keras.optimizers.RMSprop(),
             loss='binary_crossentropy',
             metrics=['acc'])
path="D:/workspace of spyder/毕业设计/my project data/model_file/pretrainning_models/"
model3.save(path+'model3.h5')


##建立model4
detctor_length=16
num_detector=32
num_hidden_unit=32
weight_decay = 0.01

model4=tf.keras.Sequential()
model4.add(layers.Conv1D(num_detector,detctor_length,input_shape=((100,4)),activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
model4.add(layers.Conv1D(num_detector,detctor_length,activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
model4.add(layers.GlobalMaxPool1D())
model4.add(layers.Dense(num_hidden_unit,activation='relu'))
model4.add(layers.Dense(num_hidden_unit,activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
model4.add(layers.Dropout(0.5))
model4.add(layers.Dense(1,activation='sigmoid'))
## 模型编译
model4.compile(optimizer=tf.keras.optimizers.RMSprop(),
             loss='binary_crossentropy',
             metrics=['acc'])
path="D:/workspace of spyder/毕业设计/my project data/model_file/pretrainning_models/"
model4.save(path+'model3.h5')



