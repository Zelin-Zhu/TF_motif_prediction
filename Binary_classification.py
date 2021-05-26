# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:11:30 2021

@author: ASUS
"""
#二分类模型
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import shutil  
import os
class Binary_classification:
    #初始化
    def __init__(self,cell_line,train_x,train_y,validate_x,validate_y,data_name):
        self.model=keras.Sequential()
        self.cell_line=cell_line
        self.train_x=train_x
        self.train_y=train_y
        self.validate_x=validate_x
        self.validate_y=validate_y
        self.history=None
        self.model_constructed=False#标记模型是否已经被建立
        self.data_name=data_name
    
    #建立模型
    def model_construct(self,detector_length=24,num_detector=32,num_hidden_unit=32,weight_decay=0.01):
        self.model.add(layers.Conv1D(num_detector,detector_length,input_shape=(self.train_x.shape[1:]),activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(layers.GlobalMaxPool1D())
        self.model.add(layers.Dense(num_hidden_unit,activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(1,activation='sigmoid'))
        self.model_constructed=True
    #模型编译
    def model_compile_and_fit(self,epochs=100,batch_size=128):
        if self.model_constructed==False:
            print("model has not been constructed!!")
        else:
            #定义callback
            
            file="D:/workspace of spyder/毕业设计/my project data/model_file/"+self.cell_line+"_"+self.data_name+"_log/"
            if os.path.exists(file):
                
                shutil.rmtree(file)  
                os.mkdir(file)          #在每次训练前先清空文件夹
            else:
                os.mkdir(file)
            checkpoint_filepath="D:/workspace of spyder/毕业设计/my project data/model_file/"+self.cell_line+"_"+self.data_name+"_log/"+"weights.{epoch:02d}-{val_acc:.2f}.hdf5"
            

           
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=False,
                monitor='val_acc',
                mode='max',
                save_best_only=True)
            model_earlystopping_callback=tf.keras.callbacks.EarlyStopping(
                    monitor="val_acc",
                    min_delta=0,
                    patience=10,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=True,
                )
            
            self.model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                               loss='binary_crossentropy',
                               metrics=['acc'])
            self.history = self.model.fit(self.train_x,self.train_y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(self.validate_x,self.validate_y),
                    callbacks=[model_checkpoint_callback,model_earlystopping_callback])
        
            plt.plot(self.history.epoch,self.history.history.get('acc'),label='acc')
            plt.plot(self.history.epoch,self.history.history.get('val_acc'),label='val_acc')
            plt.xlabel('epoches')
            plt.ylabel('accuracy')
            plt.title(self.cell_line+"_"+self.data_name)
            plt.legend()
          
            plt.savefig(file+self.cell_line+"_"+self.data_name+'_train_curve.jpg')

        
    