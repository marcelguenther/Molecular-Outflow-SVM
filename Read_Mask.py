# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:09:32 2021

@author: Phillip
"""

import pickle as pkl


def Read_Mask(__FILE_NAME__):
    ## Open the file containing the mask (reading is linewise)
    outflow_file = open(__FILE_NAME__, "rb")
    
    
    ## Reading out the sigma level
    SIGMA_LEVELL = pkl.load(outflow_file)
    SIGMA_LEVEL = float(SIGMA_LEVELL.split()[2])
    
    print(SIGMA_LEVELL)
    
    ## Jumping unimportand lines
    print(pkl.load(outflow_file))
    
    ## Reading mask
    outflow_mask = pkl.load(outflow_file)#.reshape(X1, X2, X3)
    
    ## Closing file
    outflow_file.close()
    
    # print(outflow_mask)
    
    return outflow_mask, SIGMA_LEVEL

# Read_Mask('./Data/G29.96-0.02_spw0_SiO_217104_sl-3.00_mask.pkl')
# Read_Mask('./Data/G29.96-0.02_spw2_13CO_220398_sl-3.00_mask_new.pkl')