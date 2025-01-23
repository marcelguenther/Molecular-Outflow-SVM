# -*- coding: iso-8859-1 -*-

#Database structure:
# - if len == 2 --> 1st dir, 2nd data or via keywords
# - if len == 3 --> 1st dir, 2nd data, 3rd mask or via keywords
# - if len >= 4 --> keywords required ("Parent dir", "Cube name", "Mask name")

## Import existing packages
import numpy as np
import xml.etree.ElementTree as ET                                                          ## import xml package
import platform                                                                             ## import platform package
import sys                                                                                  ## import sys package
from  distutils.util import strtobool
import multiprocessing as mp                                                                ## import multiprocessing package
from datetime import datetime, timedelta                                                    ## import datetime package
import os
import shutil
from pathlib import Path
import time

## Import self written packages
from SVM_halv import zoom
from ConcentrateFV import ConcentrateTrainFV as ctfv
from Create_Feature_Vector import create_feature_vector as cfv
from Create_Feature_Vector import check_copy_database as ccd
from defineoutflow import create_outflow_mask as com
from SVM_halv import TrainSupportVectorMachine as TSVM
from SVM_halv import ApplySupportVectorMachine as ASVM
from SVM_halv import GenerateDataSets as GDS
from read_FV import Read_Feature_Vector as rfv
from Plot_Results import PlotResults as plr

    
#if __name__ == '__main__':
def Call_Main(MasterFile, SavePath=None, ReadSVMDir="./", train=False):

    tcm = time.time()

    print("Start Main", datetime.now())
    ## Outflow Master File
    MasterFileDir = str(Path(__file__).parent.resolve()) + "/" # "./Files/"

    print(MasterFile)

    ## Read the master tree
    try:
        MasterTree = ET.parse(os.path.join(MasterFileDir+MasterFile))

    except Exception as e:
        exception_type, exception_object, exception_traceback = sys.exc_info()
        filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno
        
        print("\nReading the data from the %s file failed." %(MasterFile))
        print("Its directory was indicated as %s" %(MasterFileDir))
        print("Exception type:\t%s" %(exception_type))
        print("File name:\t%s" %(filename))
        print("Line number:\t%s" %(line_number))
        print("The error itself:\n%s\n\n" %(e))        
        print("\n\nPlease indicate a valid data file and directory.")

        sys.exit(1)


    ## Get local name
    namelocal = MasterTree.find("namelocal").text
    
    ## Give the sub roots for a local machine and a server
    if platform.node() == namelocal:
        flagroot = MasterTree.find("flags")
        pathroot = MasterTree.find("pathslocal")
        fileroot = MasterTree.find("fileslocal")
        pararoot = MasterTree.find("parameters")

        #print(pathroot)

    else:
        flagroot = MasterTree.find("flags")
        pathroot = MasterTree.find("pathsserver")
        fileroot = MasterTree.find("filesserver")
        pararoot = MasterTree.find("parameters")

    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ## initialize dictionary for model parameters
    MyParameters = {}

    ### To simulate different cube sizes
    MyParameters["lx"] = 100
    MyParameters["ly"] = 100
    MyParameters["lz"] = 100

    ## Get the given flags
    MyParameters['MASK'] = strtobool(flagroot.find("MASK").text)
    MyParameters['CREATE_FV'] = strtobool(flagroot.find("CREATE_FV").text)
    MyParameters['TRAIN_SVM'] = strtobool(flagroot.find("TRAIN_SVM").text)
    MyParameters['HALV'] = strtobool(flagroot.find("HALV").text)
    MyParameters['Plot_GS'] = strtobool(flagroot.find("Plot_GS").text)
    MyParameters['Grid_SVM'] = strtobool(flagroot.find("Grid_SVM").text)
    MyParameters['create_fits'] = strtobool(flagroot.find("create_fits").text)
    MyParameters['CHECK_FV'] = strtobool(flagroot.find("CHECK_FV").text)
    MyParameters['CHECK_FV2D'] = strtobool(flagroot.find("CHECK_FV2D").text)
    MyParameters['CHECK_FI'] = strtobool(flagroot.find("CHECK_FI").text)
    MyParameters['Plot_CM'] = strtobool(flagroot.find("Plot_CM").text)
    MyParameters['FV_CUBES'] = strtobool(flagroot.find("FV_CUBES").text)
    MyParameters['FixRandom'] = strtobool(flagroot.find("FixRandom").text)
    MyParameters['DatabaseStatistic'] = strtobool(flagroot.find("DatabaseStatistic").text)
    MyParameters['DatabaseAutomatic'] = strtobool(flagroot.find("DatabaseAutomatic").text)
    MyParameters['DatabaseManual'] = strtobool(flagroot.find("DatabaseManual").text)
    MyParameters['ProductionRun'] = strtobool(flagroot.find("ProductionRun").text)
    MyParameters['ThesisMode'] = strtobool(flagroot.find("ThesisMode").text)


    ## Read in other relevant stuff from the XML file
    ## read the paths from the Master File
    if SavePath == None:
        MyParameters['SavePath'] = pathroot.find("SavePath").text                                   ## Path to save in the results
    else:
        MyParameters["SavePath"] = SavePath

    #MyParameters['SavePath'] = SavePath                                   ## Path to save in the results
    MyParameters['ReadPath'] = pathroot.find("ReadPath").text                                   ## Path to save in the results
    MyParameters['DatabasePath'] = pathroot.find("DatabasePath").text                           ## Path to save in the results
    MyParameters['ReadFV'] = pathroot.find("Read_FV_Path").text                                 ## Path to save in the results
    #MyParameters['ReadSVM'] = pathroot.find("Read_SVM_Path").text                               ## Path to save in the results
#    if MyParameters["TRAIN_SVM"] == True:
#       MyParameters['ReadSVM'] = pathroot.find("Read_SVM_Path").text
#   elif MyParameters["TRAIN_SVM"] == False and train == True:
    MyParameters['ReadSVM'] = pathroot.find("Read_SVM_Path").text
#    else:
#        MyParameters['ReadSVM'] = ReadSVMDir


    print("ReadSVM: ", MyParameters['ReadSVM'])
    ## read in the files names from the Master File
    MyParameters['Read_SVM_name'] = fileroot.find("SVM_file").text                              ## Path to save in the results
    MyParameters['database_file'] = fileroot.find("database_file").text                         ## Path to save in the results
    MyParameters['database_name'] = ".".join(MyParameters['database_file'].split(".")[:-1])     ## Name of the database
    
    ## Subcube size
    MyParameters['SigmaSC'] = float(pararoot.find("SigmaSC").text)                          ## define size of the test data
    MyParameters['VelocitySC'] = float(pararoot.find("VelocitySC").text)                          ## define size of the test data

    ## read in the training feature values
    MyParameters['kernel'] = pararoot.find("kernel").text                                       ## the used kernel
    MyParameters['train_size'] = float(pararoot.find("train_size").text)                          ## define size of the test data
    MyParameters['SigmaLevel'] = pararoot.find("SigmaLevel").text                               ## define sigma noise level
    if MyParameters["SigmaLevel"] == "None" or MyParameters["SigmaLevel"] == "none":          ## Turn the SigmaLevel to None or a float
        MyParameters["SigmaLevel"] = None
    else:
        MyParameters['SigmaLevel'] = float(MyParameters["SigmaLevel"])

    MyParameters['grid_scoring'] = pararoot.find("grid_scoring").text                                  ## number of components
    MyParameters['gammaspacing'] = pararoot.find("gammaspacing").text                                  ## number of components
    MyParameters['gamman'] = int(pararoot.find("gamman").text)                                  ## number of components
    MyParameters['gammamin'] = float(pararoot.find("gammamin").text)                            ## define source size
    MyParameters['gammamax'] = float(pararoot.find("gammamax").text)                            ## define source size
    MyParameters['cspacing'] = pararoot.find("cspacing").text                                          ## number of components
    MyParameters['cn'] = int(pararoot.find("cn").text)                                          ## number of components
    MyParameters['cmin'] = float(pararoot.find("cmin").text)                                    ## define source size
    MyParameters['cmax'] = float(pararoot.find("cmax").text)                                    ## define source size
    MyParameters['fdmp'] = float(pararoot.find("fdmp").text)                                    ## define source size
    MyParameters['fi_scoring'] = pararoot.find("fi_scoring").text                                    ## define source size

    ## Get the number of processes
    MyParameters['npro'] = int(pararoot.find("npro").text)                                      ## define number of used processes
    MyParameters['npro'] = min(MyParameters['npro'], int(np.floor(mp.cpu_count()*.8)))          ## ensure that the number is smaller than 90% of the computer cores

    #print("Number of processes = %i\n" %(MyParameters["npro"]))
    
    ## Set the random seed if requested
    if MyParameters["FixRandom"] == True:
        np.random.seed(42)


    ## create current and all used paths
    ## Time-stamp for parent dir to create unique dirs
    currenttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ## Name all other files
    #MyParameters['NameDir'] = "%s_%s/" %(currenttime, MyParameters["database_file"][:-4])
    MyParameters['NameDir'] = "%s_%s_%s/" %(MasterFile, currenttime, MyParameters["database_file"][:-4])
    #MyParameters['OutputPath'] = MyParameters['SavePath'] + "Output/"
    MyParameters['OutputPath'] = MyParameters['SavePath']
    MyParameters['DatePath'] = MyParameters['OutputPath'] + MyParameters['NameDir']
    MyParameters['FVPath'] = MyParameters['DatePath'] + "FV/"
    MyParameters['SVMPath'] = MyParameters['DatePath'] + "SVM/"
    MyParameters['CubesPath'] = MyParameters['DatePath'] + "Cubes/"
    MyParameters['PlotPath'] = MyParameters['DatePath'] + "Plots/"
    MyParameters['OtherFilesPath'] = MyParameters['DatePath'] + "Others/"
    MyParameters['TempPath'] = MyParameters['DatePath'] + "Temp/"

    ## Clean folder, just for testing purpuses
    #if os.path.isdir(MyParameters['DatePath']):
    #    shutil.rmtree(MyParameters['DatePath'])

    ## Clean temp folder, just for testing purpuses
    #if os.path.isdir(MyParameters['TempPath']):
    #    shutil.rmtree(MyParameters['TempPath'])

    ## Creating all directories
    ## create a new directory for the output if not existing
    if not os.path.isdir(MyParameters['OutputPath']):
        Path(MyParameters['OutputPath']).mkdir(parents=True)

    ## create a new parent directory if not existing
    if not os.path.isdir(MyParameters['DatePath']):
        Path(MyParameters['DatePath']).mkdir(parents=True)

    ## create a new directory for the FV if not existing
    if not os.path.isdir(MyParameters['FVPath']):
        Path(MyParameters['FVPath']).mkdir(parents=True)

    ## create a new directory for the SVM if not existing
    if not os.path.isdir(MyParameters['SVMPath']):
        Path(MyParameters['SVMPath']).mkdir(parents=True)

    ## create a new directory for the cubes if not existing
    if not os.path.isdir(MyParameters['CubesPath']):
        Path(MyParameters['CubesPath']).mkdir(parents=True)

    ## create a new directory for the plots if not existing
    if not os.path.isdir(MyParameters['PlotPath']):
        Path(MyParameters['PlotPath']).mkdir(parents=True)

    ## create a new directory for other files if not existing
    if not os.path.isdir(MyParameters['OtherFilesPath']):
        Path(MyParameters['OtherFilesPath']).mkdir(parents=True)

    ## create a new directory for temporary files if not existing
    if not os.path.isdir(MyParameters['TempPath']):
        Path(MyParameters['TempPath']).mkdir(parents=True)

    
    ## Used files names
    MyParameters['SVM_name'] = "%s_SVM.pkl" %(MyParameters['database_name'])

    ## SVM database
    MyParameters['SVM_database'] = "%sSVM_database.csv" %(MyParameters["SVMPath"])

    ## Copy database to dir
    shutil.copyfile("%s%s" %(MyParameters["DatabasePath"], MyParameters["database_file"]),
                    "%s%s" %(MyParameters["DatePath"], MyParameters["database_file"]))

    ## Shortened database name, contains only unique values and file names
    MyParameters["database_short"] = "Database_short.csv"

    ## Ensure the database names are not identical
    if MyParameters["database_file"] == MyParameters["database_short"]:
        MyParameters["database_short"] = "Database_short_1.csv"
    
    ## Check the database format, the MASK flag and save a copy of the database to the DataPath
    print("Start Check and Copy the database!")
    ts = time.time()
    MyParameters["MASK"] = ccd(MyParameters)
    if time.time()-ts <= 50:
        print("ccd took %.3f s.\n" %(time.time()-ts))
    else:
        print("ccd took %s.\n" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))

     ## möglichst viele nan entfernen mit zoom
    #if MyParameters['MASK'] == False:
     #   zoom(MyParameters)

    ## Generate the automatic and manual datasets
    if MyParameters["MASK"] == True:
        print("Start Generate Data Sets!")
        ts = time.time()
        MyParameters = GDS(MyParameters)
        if time.time()-ts <= 50:
            print("GDS took %.3f s.\n" %(time.time()-ts))
        else:
            print("GDS took %s.\n" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))

    ## Calculate feature vectors if requested
    if MyParameters['CREATE_FV'] == True:
        ## Changing the FV read path to the FV path as one wants to use out the just created FV
        MyParameters['ReadFV'] = MyParameters['FVPath']
        #print(MyParameters["FVPath"])

        print("Start Calculating Feature Vectors!")
        ts = time.time()
        cfv(MyParameters)
        if time.time()-ts <= 50:
            print("cfv took %.3f s.\n" %(time.time()-ts))
        else:
            print("cfv took %s.\n" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))


    ## Train SVM and test it afterwards
    if MyParameters['TRAIN_SVM'] == True and MyParameters['MASK'] == True:

        ## Concentrating the Feature Vectors
        print("Start Concentrating the Featrue Vectors!")
        ts = time.time()
        ctfv(MyParameters)
        if time.time()-ts <= 50:
            print("ctfv took %.3f s.\n" %(time.time()-ts))
        else:
            print("ctfv took %s.\n" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))

        ## Changing the SVM read path to the SVM path as one wants to use out the just created SVM
        MyParameters['ReadSVM'] = MyParameters['SVMPath']

        print("Start Training a SVM!")
        ts = time.time()
        MyParameters = TSVM(MyParameters)

        if time.time()-ts <= 50:
            print("TSVM took %.3f s.\n" %(time.time()-ts))
        else:
            print("TSVM took %s.\n" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))

        ### Apply the SVM to the test data
        #print("Start Applying the SVM!")
        #ts = time.time()
        #ASVM(MyParameters, mode="test")
        #if time.time()-ts <= 50:
        #    print("ASVM took %.3f s.\n" %(time.time()-ts))
        #else:
        #    print("ASVM took %s.\n" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))

        ### Generate accuracy plots
        #ts = time.time()
        #plr(MyParameters, mode="test")
        #if time.time()-ts <= 50:
        #    print("plr took %.3f s.\n" %(time.time()-ts))
        #else:
        #    print("plr took %s" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))

    ## Apply the SVM to full cubes
    print("Start Applying the SVM!")
    ts = time.time()
    ASVM(MyParameters, mode="full")
    if time.time()-ts <= 50:
        print("ASVM_1 took %.3f s.\n" %(time.time()-ts))
    else:
        print("ASVM_1 took %s.\n" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))

    #dimF = ASVM(MyParameters)

    ## Generate accuracy plots
    print("Start Plot Results!")
    ts = time.time()
    plr(MyParameters, mode="full")
    if time.time()-ts <= 50:
        print("plr_1 took %.3f s.\n" %(time.time()-ts))
    else:
        print("plr_1 took %s.\n" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))

    """
    ## Ad an empty column at the beginning to save the accurecy score or reset it
    if "Accuracy score" in data_pd.column:
        data_pd["Accuracy score"] = 0
    else:
        data_pd = data_pd.insert(loc=0, column="Accuracy score", value=0)
    """

    ## Plot the FV in a 2d plot
    #if MyParameters["CHECK_FV"] == True:
        #CFV(MyParameters, dimF)

    ## TBD

    ## Remove the temp directory
    print("Start removing the unwanted files!")
    if os.path.isdir(MyParameters['TempPath']):
        shutil.rmtree(MyParameters['TempPath'])

    if MyParameters["ProductionRun"] == True:
        # Remove FV dir
        shutil.rmtree(MyParameters['FVPath'])

        # Remove Others dir
        shutil.rmtree(MyParameters['OtherFilesPath'])

    print("\n\nAll done.\nThe whole run took %s." %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-tcm))))

    return MyParameters['ReadSVM']
if __name__ == '__main__':
    MasterFile = "SVMMaster.xml"
    Call_Main(MasterFile)
    

