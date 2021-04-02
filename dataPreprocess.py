# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:35:41 2021

@author: ASUS
"""
import pandas as pd

class DataPreprocess:
    def __init__(self,file_path):
        self.file_path=file_path
        #导入数据，并提取有效数据段
        rawdata=pd.read_csv(file_path,header=None)
        self.data=rawdata[[0,1,2,4,9]]
        self.data.columns=['chr','left','right','strength','peak']

    ##从chip-seq实验的数据中提取固定长度的可识别序列并写入文件‘filename'¶
    def GetSequencesPositionsWithLength(self,length,file_name):
        data1=self.data.copy()
        
        for i in range(len(data1)):
            data1['left'][i]=data1['left'][i]+data1['peak'][i]-int(length/2)
            data1['right'][i]=data1['left'][i]+length-1
        data1=data1[['chr','left','right']]
        data1.to_csv(file_name,header=None,index=None)
        return data1[['chr','left','right']]
   
    
    
    ###获取背景序列区间
    def get_background_seq_positions(self):
        data=self.data.copy()
        data=data.sort_values(['chr','left'])##按照染色体和序列左端进行排序
        data.reset_index(drop=True, inplace=True)
        background_seq_data=data.copy()
        for i in range(len(data)):
            if i==0:
                background_seq_data['left'][i]=1
                background_seq_data['right'][i]=data['left'][i]-1
            else:
                background_seq_data['left'][i]=data['right'][i-1]+1
                background_seq_data['right'][i]=data['left'][i]-1
        return background_seq_data[['chr','left','right']]    
    
    
    ##从background-seq位置中提取固定长度背景序列的位置信息并将其写入到文件中
    def get_FixedLength_background_seq_positions(self,num,length,file_name):
        background_seq_positions=self.get_background_seq_positions()
        df=pd.DataFrame(data=None,columns=['chr','left','right'])
        if num<len(background_seq_positions):
            k=1
            l=num
        else:
            k=int(num/len(background_seq_positions))+1# k为在每段背景序列中提取多少个固定长度的序列
            l=len(background_seq_positions)
       
        for i in range(l):
            j=1
            center=int((background_seq_positions['right'][i]+background_seq_positions['left'][i])/2)#每段背景序列的中点
            while j<=k:
                if j%2==1:
                    chr=background_seq_positions['chr'][i]
                    left=center-length*(int(j/2)+1)
                    if left<background_seq_positions['left'][i]:
                        break 
                    else:
                        right=left+length-1
                        temp=pd.DataFrame(data=[[chr,left,right]],columns=['chr','left','right'])
                        df=df.append(temp)
                        
                if j%2==0:
                    chr=background_seq_positions['chr'][i]
                    right=center+length*(int(j/2))
                    if right>background_seq_positions['right'][i]:
                        break 
                    else:
                        left=right-length+1
                        temp=pd.DataFrame(data=[[chr,left,right]],columns=['chr','left','right'])
                        df=df.append(temp)
                j=j+1
        df.to_csv(file_name,header=None,index=None)
        return df
    ##根据强度获取对应强度阈值其在data中的位置，data根据strength降序排列,获取不大于strength的最小位置，
    ##最小值大于strength则给出最小值位置,先用顺序查找，后期可用二分优化
    def get_position_with_length(self,strength):
        length=len(self.data)
        for i in range(length):
            if self.data['strength'][i]<=strength:
                return i
            if i==length-1:
                return length-1
        
        
        
    ###获取介于low和high强度之间的数量为num长度为length的序列并写入文件file_name
    def Get_seq_positions_with_seq_strength(self,num,length,low,high,file_name):
        p=self.get_position_with_length(high)
        q=self.get_position_with_length(low)
    
      
        if (q-p)+1>=num:
            data_with_strength=self.data.loc[p:q]
            data_with_strength=data_with_strength.sample(n=num)
            data1=data_with_strength.copy()
            data1.reset_index(drop=True, inplace=True)
            for i in range(len(data_with_strength)):
                data1['left'][i]=data1['left'][i]+data1['peak'][i]-int(length/2)
                data1['right'][i]=data1['left'][i]+length-1
            
            data1[['chr','left','right']].to_csv(file_name,header=None,index=None)
            return data1[['chr','left','right']]
            
        else:
            data_with_strength=self.data.loc[p:q]
            data1=data_with_strength.copy()
            data1.reset_index(drop=True, inplace=True)
            for i in range(len(data_with_strength)):
                data1['left'][i]=data1['left'][i]+data1['peak'][i]-int(length/2)
                data1['right'][i]=data1['left'][i]+length-1
            
            data1[['chr','left','right']].to_csv(file_name,header=None,index=None)
            return data1[['chr','left','right']]



file_path='D:/workspace of spyder/毕业设计/my project/datafile./wgEncodeAwgTfbsUtaGlioblaCtcfUniPk.csv'

DP=DataPreprocess(file_path)
df=DP.GetSequencesPositionsWithLength(101,'tempfile.csv')
#df2=DP.get_FixedLength_background_seq_positions(10000, 101, 'tempbackground.csv')
df3=DP.Get_seq_positions_with_seq_strength(2000, 101, 750, 1000, 'test_750_1000.csv')


