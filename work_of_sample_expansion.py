# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 09:50:48 2021

@author: ASUS
"""

import sample_expansion as se

SE=se.sample_expansion("DNd41")
SE.expand_sample_slide()
SE.retrain_loaded_model()
SE.model.evaluate(SE.x_test,SE.y_test)