# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:41:27 2021

@author: ASUS
"""

#突变预测
import numpy as np
import tensorflow as tf
def mutation_prediction(cell_line):
    
    #导入序列数据
    cell_line=cell_line
    if cell_line=="DNd41":
        parent_file="DND41"
    else:
        parent_file=cell_line      
    seqs_750_1000=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+parent_file+"/"+cell_line+"_750_1000.npy")
    seqs=seqs_750_1000
    
    #导入motif数据
    motifs=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/predicted_motifs/"+cell_line+"_motifs.npy")
    ####
 
    ####
    #导入模型
    model_DNd41=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/DNd41_16000samples_log/weights.38-0.93.hdf5")
    model_GM12878=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/GM12878_16000samples_log/weights.14-0.91.hdf5")
    model_H1hesc=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/H1hesc_16000samples_log/weights.24-0.93.hdf5")
    model_Helas3=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/Helas3_16000samples_log/weights.33-0.89.hdf5")
    model=None
    if cell_line=='DNd41' :
        model=model_DNd41
    elif cell_line=="GM12878" :
        model=model_GM12878
    elif cell_line=="H1hesc" :
        model=model_H1hesc
    elif cell_line=="Helas3" :
        model=model_Helas3
    
    #存贮突变方式
    mutations=[]
    
    for i in range(len(motifs)):
        left=motifs[i][0]
        right=motifs[i][1]
        num=motifs[i][2]
        #定义一个len(motif)*9的数组存贮导致不可识别的突变
        # 突变A =1 突变T=2 突变C=3 突变G=4
        #删除=5
        #插入A=6 插入T=7 插入C=8 插入G=9
        if right-left+1<=16:#只对motif<=16的序列进行突变预测
            
            mutation=np.zeros((right-left+1,9))
            
            
            for j in range(left,right):
                
                #进行四种替换突变，一种不变（4种可能）
                new_seq=base_substution(seqs[num], j, 'A')
                if new_seq.any()!=0:
                    
                    new_seq=np.expand_dims(new_seq,axis=0)
                 
                    if model.predict(new_seq)<0.5:
                        mutation[j-left][0]=1
                new_seq=base_substution(seqs[num], j, 'T')
                if new_seq.any()!=0:
                    new_seq=np.expand_dims(new_seq,axis=0)
                    if model.predict(new_seq)<0.5:
                        mutation[j-left][1]=1   
                new_seq=base_substution(seqs[num], j, 'C')
                if new_seq.any()!=0:
                    new_seq=np.expand_dims(new_seq,axis=0)
                    if model.predict(new_seq)<0.5:
                        mutation[j-left][2]=1   
                new_seq=base_substution(seqs[num], j, 'G')
                if new_seq.any()!=0:
                    new_seq=np.expand_dims(new_seq,axis=0)
                    if model.predict(new_seq)<0.5:
                       
                        mutation[j-left][3]=1   
               
                
                #进行删除突变在序列远离motif的一端添加被删除的碱基（一种可能）
                new_seq=base_delete(seqs[num],j)
                if new_seq.any()!=0:
                    new_seq=np.expand_dims(new_seq,axis=0)
                    
                    if model.predict(new_seq)<0.5:
                        mutation[j-left][4]=1
    
                #进行插入突变在序列的这个位置插入四种碱基，在远离motif一段删除一个碱基（4种可能）
                if j<right:
                    new_seq=base_insert(seqs[num], j, 'A')
                    if new_seq.any()!=0:
                        new_seq=np.expand_dims(new_seq,axis=0)
                        if model.predict(new_seq)<0.5:
                            mutation[j-left][5]=1
                    new_seq=base_insert(seqs[num], j, 'T')
                    if new_seq.any()!=0:
                        new_seq=np.expand_dims(new_seq,axis=0)
                        if model.predict(new_seq)<0.5:
                            mutation[j-left][6]=1
                    new_seq=base_insert(seqs[num], j, 'C')
                    if new_seq.any()!=0:
                        new_seq=np.expand_dims(new_seq,axis=0)
                        if model.predict(new_seq)<0.5:
                            mutation[j-left][7]=1
                    new_seq=base_insert(seqs[num], j, 'G')
                    if new_seq.any()!=0:
                        new_seq=np.expand_dims(new_seq,axis=0)
                        if model.predict(new_seq)<0.5:
                            mutation[j-left][8]=1
        mutations.append(mutation)
        #输出进度
        if i % int(len(motifs)/100)==0:
            print( str(i / int(len(motifs)/100))+'%')
    print('100%')   
    file="D:/workspace of spyder/毕业设计/my project data/datafile/predicted_motifs/"
    np.save(file+cell_line+"motif_mutation_prediction.npy",mutations)
    return mutations
#碱基插入在position的后一个位置插入
def base_insert(seq,position,base): #base='A' /'T'/'C'/'G' 
     if position<0 or position>len(seq)-1:
        print('position input error')
        return [0]
     left_indent=position-0
     right_indent=len(seq)-(position+1)
     new_seq=None
     if base=='A' :
         t=[[1,0,0,0]]
     if base=='T' :
         t=[[0,1,0,0]]
     if base=='C' :
         t=[[0,0,1,0]]
     if base=='G' :
         t=[[0,0,0,1]]

     if left_indent>right_indent:
         new_seq=np.append(seq[1:position],t,axis=0)
         new_seq=np.append(new_seq,seq[position:len(seq)],axis=0)
     else:
         new_seq=np.append(seq[0:position],t,axis=0)
         new_seq=np.append(new_seq,seq[position:len(seq)-1],axis=0)
     return new_seq
         


#碱基删除
def base_delete(seq,position):
    if position<0 or position>len(seq)-1:
        print('position input error')
        return [0]
   
    left_indent=position-0
    right_indent=len(seq)-(position+1)
    new_seq=None
    if left_indent>right_indent:
        if position<len(seq)-1:
            new_seq=np.append(seq[0:position],seq[position+1:len(seq)],axis=0)
        else:
            new_seq=seq[0:position]
        new_seq=np.append(np.expand_dims(seq[position],axis=0),new_seq,axis=0)
    else:
        if position>0:
            new_seq=np.append(seq[0:position],seq[position+1:len(seq)],axis=0)
        else:
            new_seq=seq[position:len(seq)]
        new_seq=np.append(new_seq,np.expand_dims(seq[position],axis=0),axis=0)
        
    return new_seq
        
#碱基替换
def base_substution(seq,position,base):#base='A' /'T'/'C'/'G'
    new_seq=seq
    if base=='A' :
        if seq[position][0]==1:
            return np.array([0])
        elif seq[position][1]==1:
            new_seq[position][1]=0
            new_seq[position][0]=1
            return new_seq
        elif seq[position][2]==1:
            new_seq[position][2]=0
            new_seq[position][0]=1
            return new_seq
        elif seq[position][3]==1:
            new_seq[position][3]=0
            new_seq[position][0]=1
            return new_seq
        else:#这个位置是N
            for i in range(4):
                new_seq[position][i]=0
            new_seq[position][0]=1
            return new_seq
            
    if base=='T' :
        if seq[position][1]==1:
            return np.array([0])
        elif seq[position][0]==1:
            new_seq[position][0]=0
            new_seq[position][1]=1
            return new_seq
        elif seq[position][2]==1:
            new_seq[position][2]=0
            new_seq[position][1]=1
            return new_seq
        elif seq[position][3]==1:
            new_seq[position][3]=0
            new_seq[position][1]=1
            return new_seq
        else:#这个位置是N
            for i in range(4):
                new_seq[position][i]=0
            new_seq[position][1]=1
            return new_seq
    if base=='C' :
        if seq[position][2]==1:
            return np.array([0])
        elif seq[position][0]==1:
            new_seq[position][0]=0
            new_seq[position][2]=1
            return new_seq
        elif seq[position][1]==1:
            new_seq[position][1]=0
            new_seq[position][2]=1
            return new_seq
        elif seq[position][3]==1:
            new_seq[position][3]=0
            new_seq[position][2]=1
            return new_seq
        else:#这个位置是N
            for i in range(4):
                new_seq[position][i]=0
            new_seq[position][2]=1
            return new_seq
    if base=='G' :
        if seq[position][3]==1:
            return np.array([0])
        elif seq[position][0]==1:
            new_seq[position][0]=0
            new_seq[position][3]=1
            return new_seq
        elif seq[position][1]==1:
            new_seq[position][1]=0
            new_seq[position][3]=1
            return new_seq
        elif seq[position][2]==1:
            new_seq[position][2]=0
            new_seq[position][3]=1
            return new_seq
        else:#这个位置是N
            for i in range(4):
                new_seq[position][i]=0
            new_seq[position][3]=1
            return new_seq

mutation_prediction_DNd41=mutation_prediction('DNd41')
mutation_prediction_GM12878=mutation_prediction('GM12878')
mutation_prediction_H1hesc=mutation_prediction('H1hesc')
mutation_prediction_Helas3=mutation_prediction('Helas3')

