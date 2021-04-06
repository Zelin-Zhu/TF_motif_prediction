# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:10:58 2021

@author: ASUS
"""

## 建立二分类模型
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

## 2.正样本和负样本打标签并随机打乱
x_positive=np.append(seqs_0_250,seqs_250_500,axis=0)
x_positive=np.append(x_positive,seqs_500_750,axis=0)
x_positive=np.append(x_positive,seqs_750_1000,axis=0)
x_background=seqs_background[0:len(x_positive)]
y_positive=np.ones(len(x_positive))
y_background=np.zeros(len(x_background))

x=np.append(x_positive,x_background,axis=0)
y=np.append(y_positive,y_background)

index=[]
for i in range(len(x)):
    index.append(i)

np.random.shuffle(index)
x=x[index]
y=y[index]

## 3. 用一半的样本x_sample,y_sample作为和测试集，训练集和测试集的划分(3:1) 
## 另外一半样本x_test,y_test用来检验，数据集扩展方法对于模型效果的提升，验证寻找出的motif的有效性
num=int(len(x)/2)
x_sample=x[0:num]
y_sample=y[0:num]
x_test=x[num:-1]
y_test=y[num:-1]

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x_sample,y_sample)

############################建立模型#################
##1.设置网络结构参数

detctor_length=24
num_detector=32
num_hidden_unit=32
from tensorflow.keras import regularizers


weight_decay = 0.005
# kernel_regularizer=regularizers.l2(weight_decay)

##2.建立模型
model=tf.keras.Sequential()
#model.add(layers.Conv1D(num_detector,detctor_length,input_shape=(train_x.shape[1:]),activation='relu'))
model.add(layers.Conv1D(num_detector,detctor_length,input_shape=(train_x.shape[1:]),activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.GlobalMaxPool1D())
#model.add(layers.Dense(num_hidden_unit,activation='relu'))
model.add(layers.Dense(num_hidden_unit,activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

##3. 模型编译
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
             loss='binary_crossentropy',
             metrics=['acc'])

##4. 模型训练
history = model.fit(train_x,train_y,
                    epochs=35,
                    batch_size=256,
                    validation_data=(test_x,test_y))


plt.plot(history.epoch,history.history.get('acc'),label='acc')
plt.plot(history.epoch,history.history.get('val_acc'),label='val_acc')
plt.legend()

##5. 保存模型

model.save('D:/workspace of spyder/毕业设计/my project data/model_file/Binary_classification_DNd41.h5') 
# 加载模型，同时加载了模型的结构、权重等信息
loaded_model = tf.keras.models.load_model('D:/workspace of spyder/毕业设计/my project data/model_file/Binary_classification_DNd41.h5')




