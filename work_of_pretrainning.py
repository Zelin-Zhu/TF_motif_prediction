# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 20:45:48 2021

@author: ASUS
"""
import tensorflow as tf

import pretrainning as pretrainning

model1=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/pretrainning_models/model1.h5")
model2=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/pretrainning_models/model2.h5")
model3=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/pretrainning_models/model3.h5")
model4=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/pretrainning_models/model4.h5")
PT_DNd41=pretrainning.Pre_trainning("DNd41")
PT_GM12878=pretrainning.Pre_trainning("GM12878")
PT_H1hesc=pretrainning.Pre_trainning("H1hesc")
PT_Helas3=pretrainning.Pre_trainning("Helas3")


epochs=50
batch_size=128

PT_DNd41.pretrain(model1,"model1",epochs,batch_size)
PT_GM12878.pretrain(model1,"model1",epochs,batch_size)
PT_H1hesc.pretrain(model1,"model1",epochs,batch_size)
PT_Helas3.pretrain(model1,"model1",epochs,batch_size)


epochs=100
PT_DNd41.pretrain(model1,"model1",epochs,batch_size)
PT_GM12878.pretrain(model1,"model1",epochs,batch_size)
PT_H1hesc.pretrain(model1,"model1",epochs,batch_size)
PT_Helas3.pretrain(model1,"model1",epochs,batch_size)


epochs=100
PT_DNd41.pretrain(model2,"model2",epochs,batch_size)
PT_GM12878.pretrain(model2,"model2",epochs,batch_size)
PT_H1hesc.pretrain(model2,"model2",epochs,batch_size)
PT_Helas3.pretrain(model2,"model2",epochs,batch_size)


epochs=100
PT_DNd41.pretrain(model3,"model3",epochs,batch_size)
PT_GM12878.pretrain(model3,"model3",epochs,batch_size)
PT_H1hesc.pretrain(model3,"model3",epochs,batch_size)
PT_Helas3.pretrain(model3,"model3",epochs,batch_size)


epochs=100
PT_DNd41.pretrain(model3,"model4",epochs,batch_size)
PT_GM12878.pretrain(model3,"model4",epochs,batch_size)
PT_H1hesc.pretrain(model3,"model4",epochs,batch_size)
PT_Helas3.pretrain(model3,"model4",epochs,batch_size)