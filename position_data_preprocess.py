# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 10:41:33 2021

@author: ASUS
"""
import dataPreprocess as dp

#定义路径变量
file_pathDND41='D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41.csv'
file_pathGM12878='D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/GM12878.csv'
file_pathH1hesc='D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/H1hesc.csv'
file_pathHelas3='D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/Helas3.csv'

##定义类变量
DP_DND41=dp.DataPreprocess(file_pathDND41)
DP_GM12878=dp.DataPreprocess(file_pathGM12878)
DP_H1hesc=dp.DataPreprocess(file_pathH1hesc)
DP_Helas3=dp.DataPreprocess(file_pathHelas3)

##提取长度为101的序列位置信息
df_DND41=DP_DND41.GetSequencesPositionsWithLength\
    (101,'D:/workspace of spyder/毕业设计/my project data/datafile/DND41/seq_positions_DNd41.csv')
df_GM12878=DP_GM12878.GetSequencesPositionsWithLength\
    (101,'D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/seq_positions_GM12878.csv')
df_H1hesc=DP_H1hesc.GetSequencesPositionsWithLength\
    (101,'D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/seq_positions_H1hesc.csv')
df_Helas3=DP_Helas3.GetSequencesPositionsWithLength\
    (101,'D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/seq_positions_Helas3.csv')

#提取约等于1000个长度为101的背景序列位置信息
df_DND41_background=DP_DND41.get_FixedLength_background_seq_positions\
(10000, 101, 'D:/workspace of spyder/毕业设计/my project data/datafile/DND41/seq_positions_background_DNd41.csv')
df_GM12878_background=DP_GM12878.get_FixedLength_background_seq_positions\
(10000, 101, 'D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/seq_positions_background_GM12878.csv')
df_H1hesc_background=DP_H1hesc.get_FixedLength_background_seq_positions\
(10000, 101, 'D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/seq_positions_background_H1hesc.csv')
df_Helas3_background=DP_Helas3.get_FixedLength_background_seq_positions\
(10000, 101, 'D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/seq_positions_background_Helas3.csv')


## 提取每个细胞系不同强度的序列各2000条
df_DND41_750_1000=DP_DND41.Get_seq_positions_with_seq_strength(2000, 101, 750, 1000,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/DND41/seq_positions_DNd41_750_1000.csv')
df_DND41_500_750=DP_DND41.Get_seq_positions_with_seq_strength(2000, 101,500 , 750,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/DND41/seq_positions_DNd41_500_750.csv')
df_DND41_250_500=DP_DND41.Get_seq_positions_with_seq_strength(2000, 101, 250, 500,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/DND41/seq_positions_DNd41_250_500.csv')
df_DND41_0_250=DP_DND41.Get_seq_positions_with_seq_strength(2000, 101, 0, 250,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/DND41/seq_positions_DNd41_0_250.csv')
    
    
df_GM12878_750_1000=DP_GM12878.Get_seq_positions_with_seq_strength(2000, 101, 750, 1000,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/seq_positions_GM12878_750_1000.csv')
df_GM12878_500_750=DP_GM12878.Get_seq_positions_with_seq_strength(2000, 101,500 , 750,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/seq_positions_GM12878_500_750.csv')
df_GM12878_250_500=DP_GM12878.Get_seq_positions_with_seq_strength(2000, 101, 250, 500,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/seq_positions_GM12878_250_500.csv')
df_GM12878_0_250=DP_GM12878.Get_seq_positions_with_seq_strength(2000, 101, 0, 250,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/seq_positions_GM12878_0_250.csv')    
    
#############################    
df_H1hesc_750_1000=DP_H1hesc.Get_seq_positions_with_seq_strength(2000, 101, 750, 1000,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/seq_positions_H1hesc_750_1000.csv')
df_H1hesc_500_750=DP_H1hesc.Get_seq_positions_with_seq_strength(2000, 101,500 , 750,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/seq_positions_H1hesc_500_750.csv')
df_H1hesc_250_500=DP_H1hesc.Get_seq_positions_with_seq_strength(2000, 101, 250, 500,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/seq_positions_H1hesc_250_500.csv')
df_H1hesc_0_250=DP_H1hesc.Get_seq_positions_with_seq_strength(2000, 101, 0, 250,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/seq_positions_H1hesc_0_250.csv')      
    
##################################    
df_Helas3_750_1000=DP_Helas3.Get_seq_positions_with_seq_strength(2000, 101, 750, 1000,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/seq_positions_Helas3_750_1000.csv')
df_Helas3_500_750=DP_Helas3.Get_seq_positions_with_seq_strength(2000, 101,500 , 750,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/seq_positions_Helas3_500_750.csv')
df_Helas3_250_500=DP_Helas3.Get_seq_positions_with_seq_strength(2000, 101, 250, 500,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/seq_positions_Helas3_250_500.csv')
df_Helas3_0_250=DP_Helas3.Get_seq_positions_with_seq_strength(2000, 101, 0, 250,\
 'D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/seq_positions_Helas3_0_250.csv')      
    
    
