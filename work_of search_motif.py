# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:17:57 2021

@author: ASUS
"""
import numpy as np
import search_motif as sm
import tensorflow as tf

##找到seqs基于model预测的motif
class Get_motif:
    def __init__(self,model,seqs):
        self.model=model
        self.seqs=seqs
        self.predicts=self.model.predict(self.seqs)
        
    #获取seqs中a到b区间序列的motif
    def get_motif(self,a,b):
        MS=sm.Motif_Search(self.model,self.seqs)
        motifs=[]
        for i in range(a,b+1):
            motif=MS.find_motif_modified(self.seqs[i])
            motifs.append(motif)
        motifs=np.array(motifs)
        return motifs
    
    #获取所有的motif并写入文件中
    def get_motif_all(self,motif_file_name_dot_npy):
        MS=sm.Motif_Search(self.model,self.seqs)
        motifs=MS.find_motif_all(motif_file_name_dot_npy)
        return motifs
        
###############################################################################
##找到模型预测大于0.9的序列的motif并写入文件
def get_motif_of_clear_positive_train_x(GM,path_motif_of_clear_positive_train_x_dot_npy):
    motifs=[]#这里的motif包括在train_x中的位置i,和motif的左右端点p,q    motif=[p,q,i]
    for i in range(len(GM.predicts)):
        
        if GM.predicts[i]>0.9:
            t=GM.get_motif(i,i)
            motif=list(t[0])
            motif.append(i)
            motifs.append(motif)
        ##输出进度
        if i % int(len(GM.predicts)/100)==0:
                print( str(i / int(len(GM.predicts)/100))+'%')
          
    print("100%")             
    motifs=np.array(motifs)
    np.save(path_motif_of_clear_positive_train_x_dot_npy,motifs)
    return motifs

#根据细胞系和模型找到clear_positive_seqs的motif，保存到path中
def get_motifs_with_cell_line(cell_line,model,path):
    parent_file=''
    if cell_line=="DNd41":
        parent_file="DND41"
    else:
        parent_file=cell_line
    seqs_750_1000=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+parent_file+"/"+cell_line+"_750_1000.npy")
    GM=Get_motif(model,seqs_750_1000)
    motifs=get_motif_of_clear_positive_train_x(GM,path+cell_line+"_motifs.npy")
    return motifs



#导入模型
model_DNd41=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/DNd41_16000samples_log/weights.38-0.93.hdf5")
model_GM12878=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/GM12878_16000samples_log/weights.14-0.91.hdf5")
model_H1hesc=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/H1hesc_16000samples_log/weights.24-0.93.hdf5")
model_Helas3=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/Helas3_16000samples_log/weights.33-0.89.hdf5")
path="D:/workspace of spyder/毕业设计/my project data/datafile/predicted_motifs/"

#提取motifs

motifs_DNd41=get_motifs_with_cell_line("DNd41",model_DNd41,path)
motifs_GM12878=get_motifs_with_cell_line("GM12878",model_GM12878,path)
motifs_H1hesc=get_motifs_with_cell_line("H1hesc",model_H1hesc,path)
motifs_Helas3=get_motifs_with_cell_line("Helas3",model_Helas3,path)





