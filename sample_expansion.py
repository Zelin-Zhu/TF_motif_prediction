# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:34:22 2021

@author: ASUS
"""

#######利用找到的motif进行样本扩充，以提高模型准确度
#1.导入数据和模型
#2.导入motif数据并处理
##2.1提取长度小于20的motif
##2.2通过滑动motif在原序列中的位置扩增正面样本
##2.2通过随机化motif两端的无关序列扩增正面样本
#3.将利用扩增的样本和新的反面样本继续训练模型
#4.将继续训练的模型在test数据集上进行预测
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers

class sample_expansion:
    
    def __init__(self,cell_line):
        model_path="D:/workspace of spyder/毕业设计/my project data/model_file/Binary_classification_"+cell_line+".h5" 
        self.model=tf.keras.models.load_model(model_path)
        self.model_retrained=None
        self.p_file=cell_line
        if cell_line=="Dnd41": #DNd细胞系的文件夹命名出了点问题
            self.p_file="DND41"
        self.x_test=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+self.p_file+"/x_test_"+cell_line+".npy")
        self.y_test=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+self.p_file+"/y_test_"+cell_line+".npy")
        self.x_validate=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+self.p_file+"/x_validate_"+cell_line+".npy")
        self.y_validate=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+self.p_file+"/y_validate_"+cell_line+".npy")
        self.motifs_train_x=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+self.p_file+"/clear_positive_x_train_motifs_"+self.p_file+".npy")
        self.expanded_samples_slide=None
        self.expanded_samples_padding=None
        self.evaluation_loaded_model=self.model.evaluate(self.x_test,self.y_test)
        self.x_train=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+self.p_file+"/x_train_"+self.p_file+".npy")
        self.model_retrained=None
        self.background=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+self.p_file+"/"+self.p_file+"_background.npy")
    #通过滑动motif进行样本扩增
    ##  #先检查motif的长度,选择长度小于等于16的motif
        #产生连个随机数r1,r2 ,r1介于0到p之间,r2介于0到99-q之间表示向左移动r1或向右移动r2
        #根据motif是在train数据集中的位置进行移动，将新的样本加入到new_samples中
        #将new_samples赋值给self.expanded_samples
    def expand_sample_slide(self,times):
        new_samples=[]
        for j in range(times):
            for i in range(len(self.motifs_train_x)):
                left=self.motifs_train_x[i][0]
                right=self.motifs_train_x[i][1]
                position=self.motifs_train_x[i][2]
                if (right-left)+1<=16 and position<len(self.x_train):
                    position=self.motifs_train_x[i][2]
                    sample=np.zeros((len(self.x_test[0]),4))
                    
                    r1=np.random.randint(left)
                    r2=np.random.randint(len(self.x_test[0])-1-right)
                    sample[left-r1:right-r1+1,:]=self.x_train[position][left:right+1,:]  #motif段
                    sample[0:left-r1,:]=self.x_train[position][0:left-r1,:]            #最左段
                    sample[right:len(sample),:]=self.x_train[position][right:len(sample),:]#最右段                                        
                    sample[right-r1+1:right,:]=self.x_train[position][left-r1+1:left,:]  #填补段
                    new_samples.append(sample)   
                   
                    sample[left+r2:right+r2+1,:]=self.x_train[position][left:right+1,:]  #motif段
                    sample[0:left,:]=self.x_train[position][0:left,:]            #最左段
                    sample[right+r2:len(sample),:]=self.x_train[position][right+r2:len(sample),:]#最右段                                        
                    sample[left:left+r2,:]=self.x_train[position][right:right+r2,:]  #填补段
                   
                    new_samples.append(sample)
        new_samples=np.array(new_samples)
        self.expanded_samples_slide=new_samples
        return new_samples
    
    ##通过补充全A，全C全T全G序列来进行样本扩充
    
    def expand_sample_padding(self):
        new_samples=[]
        for i in range(len(self.motifs_train_x)):
            left=self.motifs_train_x[i][0]
            right=self.motifs_train_x[i][1]
            position=self.motifs_train_x[i][2]
            
            sample=np.zeros((len(self.x_train[0]),4))
            sample[:,0]=1#全部填充为A
            sample[left:right,:]=self.x_train[position][left:right,:]  #motif段
            new_samples.append(list(sample))   
            
            sample=np.zeros((len(self.x_train[0]),4))
            sample[:,1]=1#全部填充为T
            sample[left:right,:]=self.x_train[position][left:right,:]  #motif段
            new_samples.append(list(sample))   
            
            sample=np.zeros((len(self.x_train[0]),4))
            sample[:,2]=1#全部填充为C
            sample[left:right,:]=self.x_train[position][left:right,:]  #motif段
            new_samples.append(list(sample))   
            
            sample=np.zeros((len(self.x_train[0]),4))
            sample[:,3]=1#全部填充为G
            sample[left:right,:]=self.x_train[position][left:right,:]  #motif段
            new_samples.append(list(sample))       
               
        new_samples=np.array(new_samples)
        self.expanded_samples_padding=new_samples
        return new_samples
        
        
        
        
        
    def retrain_model(self):
        if len(self.expanded_samples_slide)==0:
            print("no expanded samples")
            return 0
        
        x_background=self.background[0:len(self.expanded_samples_slide)+len(self.x_train)]
        
        
        x=np.append(self.expanded_samples_slide,self.x_train,axis=0)
        print(len(x))
        x=np.append(x,x_background,axis=0)
       
        y_positive=np.ones(len(self.expanded_samples_slide)+len(self.x_train))
        y_background=np.zeros(len(x_background))
        y=np.append(y_positive,y_background)
        
        index=[]
        for i in range(len(x)):
            index.append(i)     
        np.random.shuffle(index)
        x=x[index]
        y=y[index]
        
        detctor_length=24
        num_detector=32
        num_hidden_unit=32
        from tensorflow.keras import regularizers
        
        
        weight_decay = 0.005
        # kernel_regularizer=regularizers.l2(weight_decay)
        
        ##2.建立模型
        model=tf.keras.Sequential()
        #model.add(layers.Conv1D(num_detector,detctor_length,input_shape=(train_x.shape[1:]),activation='relu'))
        model.add(layers.Conv1D(num_detector,detctor_length,input_shape=(x.shape[1:]),activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.GlobalMaxPool1D())
        #model.add(layers.Dense(num_hidden_unit,activation='relu'))
        model.add(layers.Dense(num_hidden_unit,activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1,activation='sigmoid'))
        
        model.compile(optimizer=tf.keras.optimizers.RMSprop(),
             loss='binary_crossentropy',
             metrics=['acc'])

        ##4.模型训练
        history = model.fit(x,y,
                    epochs=25,
                    batch_size=256,
                    validation_data=(self.x_validate,self.y_validate))
        plt.plot(history.epoch,history.history.get('acc'),label='acc')
        plt.plot(history.epoch,history.history.get('val_acc'),label='val_acc')
        plt.legend()
        self.model_retrained=model
        model.save("D:/workspace of spyder/毕业设计/my project data/model_file/retrained_Binary_classification_"+self.p_file+".h5") 
## 产生全ATCG的背景序列
    def generate_background(self,num):
        
        new_samples=[]
        for i in range(num):
            if i%4==0: 
                sample=np.zeros((100,4))
                sample[:,0]=1#全部填充为A
                new_samples.append(list(sample))   
            if i%4==1:
                sample=np.zeros((100,4))
                sample[:,1]=1#全部填充为T
                new_samples.append(list(sample))  
            if i%4==2: 
                sample=np.zeros((100,4))
                sample[:,2]=1#全部填充为C
                new_samples.append(list(sample))   
            if i%4==3:
                sample=np.zeros((100,4))
                sample[:,3]=1#全部填充为G
                new_samples.append(list(sample))  
            
               
        new_samples=np.array(new_samples)
        return new_samples
    
    def retrain_model2(self):
        if  len(self.expanded_samples_padding)==0:
            print("no expanded samples")
            return 0
       
        x_background=self.background[0:len(self.expanded_samples_padding)+len(self.x_train)*2]
        temp=self.generate_background(len(self.expanded_samples_padding)+len(self.x_train))
        x_background=np.append(x_background,temp,axis=0)
        
        x=np.append(self.expanded_samples_padding,self.x_train,axis=0)
        x=np.append(x,x_background,axis=0)
       
        y_positive=np.ones(len(self.expanded_samples_padding)+len(self.x_train))
        y_background=np.zeros(len(x_background))
        y=np.append(y_positive,y_background)
        
        index=[]
        for i in range(len(x)):
            index.append(i)     
        np.random.shuffle(index)
        x=x[index]
        y=y[index]
        
        detctor_length=24
        num_detector=32
        num_hidden_unit=32
        from tensorflow.keras import regularizers
        
        
        weight_decay = 0.005
        # kernel_regularizer=regularizers.l2(weight_decay)
        
        ##2.建立模型
        model=tf.keras.Sequential()
        #model.add(layers.Conv1D(num_detector,detctor_length,input_shape=(train_x.shape[1:]),activation='relu'))
        model.add(layers.Conv1D(num_detector,detctor_length,input_shape=(x.shape[1:]),activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.GlobalMaxPool1D())
        #model.add(layers.Dense(num_hidden_unit,activation='relu'))
        model.add(layers.Dense(num_hidden_unit,activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1,activation='sigmoid'))
        
        model.compile(optimizer=tf.keras.optimizers.RMSprop(),
             loss='binary_crossentropy',
             metrics=['acc'])

        ##4.模型训练
        history = model.fit(x,y,
                    epochs=50,
                    batch_size=256,
                    validation_data=(self.x_validate,self.y_validate))
        plt.plot(history.epoch,history.history.get('acc'),label='acc')
        plt.plot(history.epoch,history.history.get('val_acc'),label='val_acc')
        plt.legend()
        self.model_retrained=model
        model.save("D:/workspace of spyder/毕业设计/my project data/model_file/retrained_Binary_classification_"+self.p_file+".h5") 

SE=sample_expansion("DNd41")
SE.expand_sample_slide(2)
SE.retrain_model()

print(SE.evaluation_loaded_model)
SE.model_retrained.evaluate(SE.x_test,SE.y_test)
