# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:44:19 2021

@author: ASUS
"""
import tensorflow as tf
import numpy as np



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
    
    ##返回将序列0-p-1和q+1-len(seq)设为A的序列
    def set_seq_withA(self,seq,p,q):
        new_seq=seq.copy()
        if p>0:
            temp=np.zeros((p,4))
            for i in range(p):
                temp[i][0]=1
            new_seq[0:p]=temp
        if q<len(seq)-1:  
            temp=np.zeros((len(seq)-1-q+1,4))
            for i in range(len(seq)-1-q+1):
                temp[i][0]=1
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
    def find_motif_modified(self,seq):
        predict=self.model.predict(np.expand_dims(seq,axis=0))
        p=0
        q=len(seq)-1
        flag=1 #用来标记两边都不能进一步压缩
        while  flag==1:
            mid=int((p+q)/2)
            half=int((q-p)/2)
            seq_left=self.set_seq_withA(seq, p, mid)
            seq_right=self.set_seq_withA(seq, mid, q)
            seq_center=self.set_seq_withA(seq, p+half, q-half)
            predict_left=self.model.predict(np.expand_dims(seq_left,axis=0))
            predict_right=self.model.predict(np.expand_dims(seq_right,axis=0))
            predict_center=self.model.predict(np.expand_dims(seq_center,axis=0))
            if abs(predict_left-predict)/predict<0.1:
                q=mid
            elif abs(predict_right-predict)/predict<0.1:
                p=mid
            elif abs(predict_center-predict)/predict<0.1:
                p=p+half
                q=q-half
            else:
                while flag==1:
                    new_seq1=self.set_seq_withA(seq, p+1, q)
                    new_seq2=self.set_seq_withA(seq, p, q-1)
                    new_seq3=self.set_seq_withA(seq, p+1, q-1)
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
    
    ##找到所有序列的motif并写入文件motif_file_name.npy
    def find_motif_all(self,motif_file_name_dot_npy):
        motifs=np.zeros((len(self.sequences),2))
        for i in range(len(self.sequences)):
            if i % int(len(self.sequences)/10)==0:
                print( str(i / int(len(self.sequences)/10)*10)+'%')
            find=self.find_motif_modified(self.sequences[i])
            motifs[i][0]=find[0]
            motifs[i][1]=find[1]
        print('100.0%')
        np.save(motif_file_name_dot_npy,motifs)
        return motifs
# MS=Motif_Search(loaded_model, seqs_750_1000)
# motif=MS.find_motif_all('D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_motifs_seqs_750_1000.npy')
  
#test=np.load('D:/workspace of spyder/毕业设计/DNd41_motifs_seqs_750_1000.npy')
           
       
    