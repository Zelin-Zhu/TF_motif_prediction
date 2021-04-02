# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:44:19 2021

@author: ASUS
"""
import tensorflow as tf
import numpy as np

loaded_model = tf.keras.models.load_model('model1_classification.h5')
seqs_750_1000=np.load("D:/workspace of spyder/毕业设计/my project/datafile/DNd41_750_1000.npy")
predicts=loaded_model.predict(seqs_750_1000)


class Motif_Search:
    def __init__(self,model,sequences):
        self.model=model
        self.sequences=sequences
        
    ## 返回将seq序列0-p,q-len(seq)进行随机化的序列
    def randomize_seq(self,seq,p,q):
        new_seq=seq.copy()
        if p>0:
            temp=np.zeros((p,4))
            for i in range(p):
                temp[i][np.random.randint(4)]=1
            new_seq[0:p]=temp
        if q<len(seq)-1:  
            temp=np.zeros((len(seq)-1-q+1,4))
            for i in range(len(seq)-1-q+1):
                temp[i][np.random.randint(4)]=1
            new_seq[q:len(seq)]=temp
        return new_seq
    
    ##找到一个序列的motif区间(p,q) 经过randomize_seq随机化后模型预测改变值变化小于10% 
    def find_motif(self,seq):
        predict=self.model.predict(np.expand_dims(seq,axis=0))
        p=0
        q=len(seq)-1
        flag=1 #用来标记两边都不能进一步压缩
        while  flag==1:
            new_seq1=self.randomize_seq(seq, p+1, q)
            new_seq2=self.randomize_seq(seq, p, q-1)
            new_seq3=self.randomize_seq(seq, p+1, q-1)
            new_predict1=self.model.predict(np.expand_dims(new_seq1,axis=0))
            new_predict2=self.model.predict(np.expand_dims(new_seq2,axis=0))
            new_predict3=self.model.predict(np.expand_dims(new_seq3,axis=0))
            if abs(new_predict3-predict)/predict<0.1:
                p=p+1
                q=q-1
            elif abs(new_predict1-predict)/predict<0.1:
                p=p+1
            elif abs(new_predict2-predict)/predict<0.1:
                q=q-1
            else:
                flag=0
        return [p,q]  
    def find_motif_all(self):
        motifs=np.zeros((len(self.sequences),2))
        for i in range(len(self.sequences)):
            find=self.find_motif(self.sequences[i])
            motifs[i][0]=find[0]
            motifs[i][1]=find[1]
        return motifs
MS=Motif_Search(loaded_model, seqs_750_1000)


for i in range(10):
    motif=MS.find_motif(seqs_750_1000[i])
    print(motif)

           
       
    