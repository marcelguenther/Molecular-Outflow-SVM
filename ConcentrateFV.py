
import os
import numpy as np
import pandas as pd
import pickle as pkl
import psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

def ConcentrateTrainFV(CalcParameters):

    print("Start concentrating training data.")

    ## Read the dataset and create a copy of it
    data_pd = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]), index_col=0)
    data_pd_test = data_pd
    data_pd_test["Parent dir"] = CalcParameters["TempPath"]

    ### Create an empyt lists to buffer in the concentrated FV names and their number of data points
    cFVnsheader = ["Concentrated name", "Number of vectors"]
    cFVns = list()

    ## Concentrated FV file index
    cfvi = 1

    ## Initiate the concentrated arrays
    train_con = pd.DataFrame()


    ## Train a SVM for each FV set
    for index, row in data_pd.iterrows():

        if not index+1 == len(data_pd.index):
            print("Concentrating the FV-set Nr. %i out of %i." %(index+1, len(data_pd.index)), end="\r")
        else:
            print("Concentrating the FV-set Nr. %i out of %i." %(index+1, len(data_pd.index)))
    
        ## Set the rng TBD
        rng = np.random.RandomState(0)

        ## Get the directory name of the cube and mask as well as their names
        Parent_Dir = row["Parent dir"]
        Cube_Name = row["Cube name"]
        Mask_Name = row["Mask name"]

        ## Get the file size and avilible memory in bytes
        FV_memory = os.stat("%s%s_FV.pkl" %(CalcParameters["ReadFV"], ".".join(Cube_Name.split(".")[:-1]))).st_size
        free_memory = psutil.virtual_memory().available

        ## If the free_memory gets short, save the file to disc and reset the arrays
        if FV_memory*3 >= free_memory:

            print("The availible memory is about to be exceeded. Save the file for the %i-th time." %(cfvi))

            cFVns = np.append((cFVns, ("Combined_Training_Train_Set_%i.csv" %(cfvi), len(train_con.index))))

            train_con.index = np.linspace(1, len(train_con.index), len(train_con.index), dtype=int)

            ## save the array
            train_con.to_csv("%sCombined_Training_Train_Set_%i.csv" %(CalcParameters["TempPath"], cfvi))

            ## Increasing count and reseting data frame
            cfvi += 1
            train_con = pd.DataFrame()

        ## Reading the FV (and get its header in 1st run)
        if index == data_pd.first_valid_index():
            FV_dset = pd.read_pickle("%s%s_FV.pkl" %(CalcParameters["ReadFV"], ".".join(Cube_Name.split(".")[:-1])))
            FV_header = FV_dset.columns[4:-1]
            FV_dset = FV_dset.to_numpy()
            #print(FV_header, FV_dset.shape)
 
        else:
            FV_dset = pd.read_pickle("%s%s_FV.pkl" %(CalcParameters["ReadFV"], ".".join(Cube_Name.split(".")[:-1]))).to_numpy()

        ## Drop all rows containing nans
        #FV_dset = FV_dset[~np.isnan(FV_dset).any(axis=1)]

        #if not CalcParameters["SigmaLevel"] == None:
            #print(FV_dset.T[3])
            #print(np.greater(FV_dset.T[3], CalcParameters["SigmaLevel"]))
            ## Drop all rows below sigma threshold
            #FV_dset = FV_dset[np.greater(FV_dset.T[3], CalcParameters["SigmaLevel"])]

        ## Gain the features
        X = FV_dset[:,5:]
        y = np.array(FV_dset[:,4], dtype=int)

        ## Splitting the data into test and training set and free memory space
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=CalcParameters["train_size"], random_state=rng, stratify=y)
      
        X, y = 0, 0

        #take the sigma filter from the Train data

        drei_sigma = X_train[:,-1]

        X_train = X_train[:,:-1]

        ### X_train still has all data and the sigma filter can now be used ###

        for i in range(X_train.shape[1]):
            X_train[:,i][drei_sigma == False] = np.nan
            
        nan_indices = np.any(np.isnan(X_train), axis=1)

        y_train = y_train[~nan_indices]

        X_train = X_train[~np.isnan(X_train).any(axis=1)]

        ### delet the sigma filter from the test Data ###

        X_test = X_test[:,:-1]

        ## Save the test data to file, free memory space, and update the test data base
        test_data = pd.DataFrame(data=np.concatenate((y_test.reshape(-1,1), X_test), axis=1), columns=FV_header)
        X_test, y_test = 0, 0

        test_data.to_csv("%sTraining_Test_Set_%s.csv" %(CalcParameters["TempPath"], index))

        test_data = 0

        #data_pd_test["Cube name"][index] = "Training_Test_Set_%s.csv" %(index)
        data_pd_test.loc[index, "Cube name"] = "Training_Test_Set_%s.csv" %(index)

        ## Concentrate the training data and free memory space
        train_data = pd.DataFrame(data=np.concatenate((y_train.reshape(-1,1), X_train), axis=1), columns=FV_header)
        X_train, y_train = 0, 0

        if train_con.empty:
            train_con = train_data

        else:
            train_con = pd.concat([train_con, train_data])
    
        train_data = 0

    ## Update file names and save results to file
    cFVns = np.append(cFVns, ("Combined_Training_Train_Set_%i.csv" %(cfvi), len(train_con.index)))
    cFVns = pd.DataFrame(data=cFVns.reshape(cfvi,2), columns=cFVnsheader)
    cFVns.to_csv("%sConcentrated_FV_Names.csv" %(CalcParameters["TempPath"]))

    ## Save the name list to file
    train_con.index = np.linspace(1, len(train_con.index), len(train_con.index), dtype=int)
    train_con.to_csv("%sCombined_Training_Train_Set_%i.csv" %(CalcParameters["TempPath"], cfvi))

    ## Save the test data base to the temporary data base
    data_pd_test.to_csv("%sReduced_Test_DataBase.csv" %(CalcParameters["TempPath"]))

    ## Return nothing
    return
