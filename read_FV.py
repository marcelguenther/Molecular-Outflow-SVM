# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:42:11 2021

@author: Phillip
"""

import numpy as np
import pickle as pkl


def Read_Feature_Vector(FV_file_name):
    
    ## Open the file containing the mask (reading is linewise)
    FV_file = open(FV_file_name, "rb")
    
    ## Reading out the sigma level
    header = pkl.load(FV_file)

    X1 = int(pkl.load(FV_file).split()[2])
    X2 = int(pkl.load(FV_file).split()[2])
 
    FV = np.empty([X1, X2])
    
    FV = pkl.load(FV_file).reshape(X1, X2)

    return FV
