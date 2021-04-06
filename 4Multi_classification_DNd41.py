# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:04:32 2021

@author: ASUS
"""

##多分类模型
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
######################数据准备#########################
## 1.获取数据





seqs_500_750=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_500_750.npy")
seqs_750_1000=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_750_1000.npy")
seqs_250_500=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_250_500.npy")
seqs_background=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_background.npy")
seqs_0_250=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_0_250.npy")

## 2.对各类样本进行one-hot编码


y_0_250=np.zeros((len(seqs_0_250),2))
y_0_250[:,0]=1

y_750_1000=np.zeros((len(seqs_750_1000),2))
y_750_1000[:,1]=1




x=np.append(seqs_0_250,seqs_750_1000,axis=0)



y=np.append(y_0_250,y_750_1000,axis=0)

index=[]
for i in range(len(x)):
    index.append(i)

np.random.shuffle(index)
x=x[index]
y=y[index]

##数据集划分

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y)

############################建立模型#################
##1.设置网络结构参数

detctor_length=16
num_detector=32
num_hidden_unit=32
from tensorflow.keras import regularizers


weight_decay = 0.01
# kernel_regularizer=regularizers.l2(weight_decay)

##2.建立模型
model=tf.keras.Sequential()

model.add(layers.Conv1D(num_detector,detctor_length,input_shape=(train_x.shape[1:]),activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(num_hidden_unit,activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(num_hidden_unit,activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(2,activation='softmax'))

##3. 模型编译
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
             loss='categorical_crossentropy',
             metrics=['acc'])

##4. 模型训练
history = model.fit(train_x,train_y,
                    epochs=300,
                    batch_size=256,
                    validation_data=(test_x,test_y))


plt.plot(history.epoch,history.history.get('acc'),label='acc')
plt.plot(history.epoch,history.history.get('val_acc'),label='val_acc')
plt.legend()

##5. 保存模型

model.save('D:/workspace of spyder/毕业设计/my project data/model_file/2multi_classification_DNd41.h5') 
# 加载模型，同时加载了模型的结构、权重等信息
loaded_model = tf.keras.models.load_model('D:/workspace of spyder/毕业设计/my project data/model_file/2multi_classification_DNd41.h5')
