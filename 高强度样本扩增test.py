# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:35:12 2021

@author: ASUS
"""

#在高强度数据下的样本扩增检验
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

seqs_750_1000=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_750_1000.npy")
seqs_background=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_background.npy")
seqs_background=seqs_background[0:3000]

x=seqs_750_1000
x=np.append(x,seqs_background,axis=0)
y_positive=np.ones(len(seqs_750_1000))
y_negtive=np.zeros(len(seqs_background))
y=np.append(y_positive,y_negtive,axis=0)

index=[]
for i in range(len(x)):
    index.append(i)

np.random.shuffle(index)
x=x[index]
y=y[index]
num=1000
x_sample=x[0:num]
y_sample=y[0:num]
x_test=x[num:-1]
y_test=y[num:-1]

train_x=x_sample[0:750]
validate_x=x_sample[750:1000]
train_y=y_sample[0:750]
validate_y=y_sample[750:1000]


detctor_length=24
num_detector=32
num_hidden_unit=32
from tensorflow.keras import regularizers


weight_decay = 0.01
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

##4.模型训练
history = model.fit(train_x,train_y,
                    epochs=60,
                    batch_size=256,
                    validation_data=(validate_x,validate_y))


plt.plot(history.epoch,history.history.get('acc'),label='acc')
plt.plot(history.epoch,history.history.get('val_acc'),label='val_acc')
plt.legend()

import search_motif as sm

index_of_positive_in_train=np.argwhere(train_y>0)
index_of_positive_in_train=np.squeeze(index_of_positive_in_train)
train_x_positive=train_x[index_of_positive_in_train]
SM=sm.Motif_Search(model,train_x_positive)
motifs=SM.find_motif_all("temp_motif.npy")


def expand_sample_padding(motifs,train_x_positive):
    new_samples=[]
    for i in range(len(motifs)):
        left=int(motifs[i][0])
        right=int(motifs[i][1])
        position=i
        
        sample=np.zeros((len(train_x_positive[0]),4))
        sample[:,0]=1#全部填充为A
        sample[left:right,:]=train_x_positive[position][left:right,:]  #motif段
        new_samples.append(list(sample))   
            
        sample=np.zeros((len(train_x_positive[0]),4))
        sample[:,1]=1#全部填充为T
        sample[left:right,:]=train_x_positive[position][left:right,:]  #motif段
        new_samples.append(list(sample))   
            
        sample=np.zeros((len(train_x_positive[0]),4))
        sample[:,2]=1#全部填充为C
        sample[left:right,:]=train_x_positive[position][left:right,:]  #motif段
        new_samples.append(list(sample))   
            
        sample=np.zeros((len(train_x_positive[0]),4))
        sample[:,3]=1#全部填充为G
        sample[left:right,:]=train_x_positive[position][left:right,:]  #motif段
        new_samples.append(list(sample))   
                             
    new_samples=np.array(new_samples)
    return new_samples

new_samples=expand_sample_padding(motifs, train_x_positive)
y_new_samples=np.ones(len(new_samples))

train_x_expand=np.append(train_x,new_samples,axis=0)
train_y_expand=np.append(train_y,y_new_samples,axis=0)


new_model=tf.keras.Sequential()
#model.add(layers.Conv1D(num_detector,detctor_length,input_shape=(train_x.shape[1:]),activation='relu'))
new_model.add(layers.Conv1D(num_detector,detctor_length,input_shape=(train_x.shape[1:]),activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
new_model.add(layers.GlobalMaxPool1D())
#model.add(layers.Dense(num_hidden_unit,activation='relu'))
new_model.add(layers.Dense(num_hidden_unit,activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
new_model.add(layers.Dropout(0.5))
new_model.add(layers.Dense(1,activation='sigmoid'))

##3. 模型编译
new_model.compile(optimizer=tf.keras.optimizers.RMSprop(),
             loss='binary_crossentropy',
             metrics=['acc'])

##4.模型训练
new_history = new_model.fit(train_x_expand,train_y_expand,
                    epochs=60,
                    batch_size=256,
                    validation_data=(validate_x,validate_y))


plt.plot(new_history.epoch,new_history.history.get('acc'),label='acc')
plt.plot(new_history.epoch,new_history.history.get('val_acc'),label='val_acc')
plt.legend()




