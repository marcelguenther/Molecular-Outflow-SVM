# -*- coding: iso-8859-1 -*-
'''
TBD:
    - better handling of the edges
'''

"""
Created on Thu Jun 17 16:39:25 2021

@author: Phillip, Marcel
"""

##############################################################################
### Include some packages

## Import existing packages
import astropy.io.fits as fits
from scipy.ndimage import convolve
from astropy.stats import sigma_clip
import itertools
from datetime import datetime as datetime
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from skimage.filters import difference_of_gaussians as DoG
from spectral_cube import SpectralCube as sc
from spectral_cube.utils import SpectralCubeWarning
import pandas as pd
import psutil
import sys
from tqdm import tqdm
import time
import warnings
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
import bettermoments as bm

## Import self written packages
#from FE import Feature_Extractor as fe
from Fitgaus import fit_gaus as fg
from Read_Mask import Read_Mask as rm

def Feature_Extractor(CalcParameters, FILE_CUBE, FILE_MASK="NoMask", i_set=-1):
    warnings.filterwarnings("ignore")    
    ## Try to derive the feature vectors
    print(FILE_CUBE)
    try:
        ## Read out data
        header = fits.getheader(FILE_CUBE)

        #make sure the beam is included in the header

        if 'BMAJ' not in header or 'BMIN' not in header or 'BPA' not in header:
            cube = sc.read(FILE_CUBE)
            common_beam = cube.beams.common_beam()
            cb_cube = cube.convolve_to(common_beam)
            header = cb_cube.header

        data = sc.read(FILE_CUBE).to("K").hdulist[0].data    

        ## to check if there are pixels with only nan value, do a zoom ##
        def zoom(data):
            z = 0
            data_new=data
            while np.count_nonzero(~np.isnan(data)) == np.count_nonzero(~np.isnan(data_new)):
            # NaN-Werte vorhanden: Pixel um eins verkleinern
                z += 1
                x = z - 1
                data_new = data[:, z:-z, z:-z]
                data_save = data[:, x:-x, x:-x]
            print('Größe von', data.shape,'auf', data_save.shape,'reduziert')
            return data_save
        
        #to check if there are only nan channels, remove them
        def remove_nan_channels(data):
            # Überprüfen, ob die Daten NaN-Werte enthalten
            nan_indices = np.all(np.isnan(data), axis=(1, 2))  # Indizes der Kanäle mit NaN-Werten finden
            non_nan_data = data[~nan_indices]  # Kanäle ohne NaN-Werte auswählen
            print('Größe von', data.shape,'auf', non_nan_data.shape,'reduziert')
            return non_nan_data
         
        #if np.count_nonzero(~np.isnan(data)) == np.count_nonzero(~np.isnan(data[:, 2:-2, 2:-2])):
            #data = zoom(data)

        if np.any(np.all(np.isnan(data), axis=(1, 2))):
            data = remove_nan_channels(data)

        ## Avoid arrays only containing nans or infs
        if not np.isnan(data).all() or np.isinf(data).all():

            ## Create original copy of data
            data_o = np.copy(data)

            ## Shape of data cube
            lz, lx, ly = np.shape(data)


            ## Read out mask
            if FILE_MASK != "NoMask":
                mask = sc.read(FILE_MASK).hdulist[0].data
        
            else:
                mask = np.zeros_like(data)

            ## Calculate std of the background using sigma clipping
            data_sc = sigma_clip(data, sigma=3, maxiters=None, cenfunc="mean") # , axis=0)
            scstd = np.nanstd(data_sc) # , axis=0)
            
            ### save 3 Sigma Filter in array ###
            data[data < 0] = 0
            drei_sigma_filter = data > 3 * scstd#

            ## do arcsinh on the data ## 
            data = np.arcsinh(data)

            #load information from the header#
            cdelt3 = np.abs(header['CDELT3'])
            cunit3 = header['CUNIT3']
            cdelt1 = np.abs(header['CDELT1'])
            cdelt2 = np.abs(header['CDELT2'])
            BMAJ = np.abs(header['BMAJ'])


            ## Subcube extend in spatial dimension; defined via 2 times the beam size
            npsc1 = max(int(np.ceil(BMAJ*CalcParameters["SigmaSC"]*0.5/cdelt1 - 0.5)) * 2 + 1, 3)
            npsc2 = max(int(np.ceil(BMAJ*CalcParameters["SigmaSC"]*0.5/cdelt2 - 0.5)) * 2 + 1, 3)

            ## Subcube extend in velocity dimension; defined via 25 km/s
            if cunit3 == "kms-1" or cunit3 == "km/s" or cunit3 == "km s-1" or cunit3 == "km / s":
                nvsc = max(int(np.ceil(CalcParameters["VelocitySC"]*0.5/cdelt3 - 0.5)) * 2 + 1, 20)
            elif cunit3 == "ms-1" or cunit3 == "m/s" or cunit3 == "m s-1" or cunit3 == "m / s":
                nvsc = max(int(np.ceil(CalcParameters["VelocitySC"]*1000*0.5/cdelt3 - 0.5)) * 2 + 1, 20)
            else:
                print("The velocity unit is unknown. Assuming it is in km/s.")
                nvsc = max(int(np.ceil(CalcParameters["VelocitySC"]*0.5/cdelt3 - 0.5)) * 2 + 1, 20)

            ## Place-holder for the signal-to-noise value sigma
            sigma = np.zeros_like(data, dtype=int)

            ## Create enlarged x, y, z arrays
            x = np.linspace(0,lx-1,lx, dtype=int)
            y = np.linspace(0,ly-1,ly, dtype=int)
            z = np.linspace(0,lz-1,lz, dtype=int)
            zm, xm, ym = np.meshgrid(z, x, y, indexing="ij")

            ## Prepare DoG
            DoGarrayint = np.zeros_like(data)

            for i in np.linspace(-(nvsc-1)/2, (nvsc-1)/2, nvsc, dtype=int):
                DoGarrayint[max(i, 0):min(lz+i, lz)] += data[max(-i, 0):min(lz-i, lz)]

            DoGarrayint = np.nan_to_num(DoGarrayint, nan=0)

            ## Calculating DoG
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                DoGarray = DoG(DoGarrayint, 1*BMAJ/max(cdelt1,cdelt2), 2*BMAJ/max(cdelt1,cdelt2))

            #calculate the fwhm and sum for every pixel in the range of 25km/s

            def sum_fwhm(data):
                nz, ny, nx = data.shape
                fwhm_values = np.zeros((nz, ny, nx))
                neighborhood_sum = np.zeros((nz, ny, nx))

                #all data below 0 is 0
                data[data < 0] = 0

                ## range should be 25km/s
                if cunit3 == "ms-1" or cunit3 == "m/s" or cunit3 == "m s-1" or cunit3 == "m / s":
                    Channels = 25000 / cdelt3
                    v = cdelt3 / 1000
                elif cunit3 == "kms-1" or cunit3 == "km/s" or cunit3 == "km s-1" or cunit3 == "km / s":
                    Channels = 25 / cdelt3
                    v = cdelt3
                else:
                    Channels = 25 / cdelt3
                    v = cdelt3

                Bereich = int(np.ceil(Channels)/2) - 1
    
                for k in range(nz):
                    neighborhood_area = data[max(k - Bereich, 0):min(k + Bereich, nz), :, :]  # Extract the neighborhood area around the current point for all i and j
                    neighborhood_sum[k] = np.nansum(neighborhood_area, axis=0) / len(neighborhood_area)
                    # Check if the neighborhood area contains any NaN values
                    if np.isnan(neighborhood_area).all():
                    # If all values are NaN, skip this iteration
                        continue
        
                    peak_max_fwhm_values = np.nanmax(neighborhood_area, axis=0)  # Maximum values across the area
    
                    # Find the indices of non-NaN maximum values
                    peak_max_index = np.where(neighborhood_area == peak_max_fwhm_values, 1, 0)
                    peak_max_index = np.argmax(peak_max_index, axis=0)
        
                    half_max = 0.3 * peak_max_fwhm_values  # 0.3 of the maximum value
        
                    left_idx = peak_max_index.copy()  # Copy of the maximum indices
                    right_idx = peak_max_index.copy()  # Copy of the maximum indices

                    # Find the indices where values drop below 0.3 of the maximum
                    for i in range(ny):
                        for j in range(nx):
                            while left_idx[i, j] > 0 and neighborhood_area[left_idx[i, j], i, j] > half_max[i, j]:
                                left_idx[i, j] -= 1
                            while right_idx[i, j] < len(neighborhood_area) - 1 and neighborhood_area[right_idx[i, j], i, j] > half_max[i, j]:
                                right_idx[i, j] += 1
        
                    # Calculate the FWHM values for each point
                    fwhm = right_idx - left_idx
            
                    fwhm_values[k, :, :] = fwhm * v


                return neighborhood_sum, fwhm_values

            Moment0, FWHM = sum_fwhm(data)
            
            

            ### Calculate the feature
         
            ### Feature 1
            feature_1 = DoGarray * np.sqrt(Moment0) * FWHM**2 


            ### MinMaxScaler ###
            ### possible to add another value as the maximum to include or exlude more voxels
             
            def minmax_scaler(data):
                filter = data[drei_sigma_filter] 
                #if np.nanmax(filter) > 1000:
                data_scaled = (data - np.nanmin(filter))/(np.nanmax(filter)-np.nanmin(filter))
                #else:
                 #   data_scaled = (data - np.nanmin(filter))/(1000-np.nanmin(filter))
                return data_scaled
            
            feature_1 = minmax_scaler(feature_1)

            ### Produce Feature Vectors out of these arrays
            nfv = lx*ly*lz
            xm_l = xm.reshape(nfv)
            ym_l = ym.reshape(nfv)
            zm_l = zm.reshape(nfv)
            si_l = sigma.reshape(nfv)
            ma_l = mask.reshape(nfv)
            f1_l = feature_1.reshape(nfv)
            drei_sigma = drei_sigma_filter.reshape(nfv)
            data = np.array((zm_l, xm_l, ym_l, si_l, ma_l, f1_l, drei_sigma))

            header = ["z position", "x position", "y position", "Sigma Value", "OF Pixel","Feature 5","drei_sigma"]

            df = pd.DataFrame(data=data.T, columns=header, index=None)
    
            return df

        ## Return warning if the cube is invalid
        else:
            print("\nWarning: The array %i contains just nans. It will be ignored for further calculations and removed from the database." %(i_set))
        
            return False


    ## Return warning if the cube is invalid
    except Exception as e:
        exception_type, exception_object, exception_traceback = sys.exc_info()
        filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno
        
        print("\nWarning: Failed to derive Feature Vectors from array %i. It will be ignored for further calculations and removed from the database." %(i_set))
        print("Exception type:\t%s" %(exception_type))
        print("File name:\t%s" %(filename))
        print("Line number:\t%s" %(line_number))
        print("The error itself:\n%s\n\n" %(e))

        return False
##############################################################################
### Function that checks if the database is valid and if it contains a mask, saves a copy afterwards
def check_copy_database(CalcParameters):

    ## Read the database header
    database = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_file"]), index_col=False)

    ## Check if mandatory column entries are in the header
    if "Parent dir" in database.columns and "Cube name" in database.columns:
        
        ## Drop unnamed lines
        database.drop(database.filter(regex="Unname").columns, axis=1, inplace=True)


        ## If the Mask name entry doesn't exist, set the MASK flag to False
        if "Mask name" not in database.columns and CalcParameters["MASK"] == True:
            print("Warning, the database has no information on the masks. Please check the database.\nTurning the MASK flag to 'False'.")
            CalcParameters["MASK"] = False

    ## In case of an invalid header --> count the number of columns and set a matching header
    else:

        ## 2 columns --> dir, cube
        if len(database.columns) == 2:
            header = ["Parent dir", "Cube name"]
            
            ## Turn the MASK flag to False if it's True
            if CalcParameters["MASK"] == True:
                print("Warning, the database has no information on the masks. Please check the database.\nTurning the MASK flag to 'False'.")
                CalcParameters["MASK"] = False
                
        ## 3 columns --> dir, cube, mask
        elif len(database.columns) == 3:
            header = ["Parent dir", "Cube name", "Mask name"]

        ## else the database has a wrong format
        else:
            error_text_line_1 = "The database header doesn't contain the two requested arguements 'Parent dir' and 'Cube name' and, therefore, is invalid.\n"
            error_text_line_2 = "In case the database would have a header, it would be:\n%s\n" %(database.columns)
            error_text_line_3 = "Furthermore, the database has an invalid shape. For non-header data, only data sets with 2 or 3 columns are accepted.\n"
            error_text_line_4 = "This database has %i columns.\n" %(len(database.columns))
            error_text_line_5 = "Please correct the database and/or its header try again.\n"

            error_text = error_text_line_1 + error_text_line_2 + error_text_line_3 + error_text_line_4 + error_text_line_5
            
            raise Exception(error_text)

        ## Read in the data but this time with a header
        database = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_file"]), columns=header, index_col=False)

    ## Drop duplicates
    database = database.drop_duplicates()

    ## Drop lines containing the trainings/a date
    database.drop(database.filter(regex="Date").columns, axis=1, inplace=True)


    ## Drop constant values excluding model/dir/cube/mask names
    rel_index = (database != database.iloc[0]).any()

    rel_index["Parent dir"] = True
    rel_index["Cube name"] = True
    
    if "Mask name" in database.columns:
        rel_index["Mask name"] = True

    if "FV name" in database.columns:
        rel_index["FV name"] = True


    database = database.loc[:, rel_index]

    ## Save a copy of the database in the DatePath
    database.to_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]))

    return CalcParameters["MASK"]


###############################################################################
### Function that calculates the features of all pixels
def create_feature_vector(CalcParameters, data_set=None):
    
    ## Generate FVs for the whole database if no cube is declared
    if data_set is None:

        ## Read the database
        data_pd = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]), index_col=0)

        Parent_Dirs = data_pd["Parent dir"]
        Cube_Names = data_pd["Cube name"]

        if "Mask name" in data_pd.columns and CalcParameters["MASK"] == True:
            Mask_Names = data_pd["Mask name"]

        ## Add a column for the FV dirs and names, if not existing
        if not "FV name" in data_pd.columns:
            data_pd["FV name"] = ""


    ## Or use the handed over data set
    else:
        Parent_Dirs = data_set["Parent dir"]
        Cube_Names = data_set["Cube name"]
        FV_name = data_set["FV name"]

        if "Mask name" in data_set.columns and CalcParameters["MASK"] == True:
            Mask_Names = data_set["Mask name"]

    ## List to catch the indexes of failed feature extraction tries
    inval_cubes = []

    ## Loop over all data sets
    for i_set in range(len(Parent_Dirs)):

        ## Return a status update
        if not i_set+1 == len(Parent_Dirs):
            print("Creating the FV of set %i of %i." %(i_set+1, len(Parent_Dirs)), end="\r")
        else:
            print("Creating the FV of set %i of %i." %(i_set+1, len(Parent_Dirs)))

        ## Get the cube name inkl its dir
        Parent_Dir = Parent_Dirs[i_set]
        Cube_Name = Cube_Names[i_set]
        FILE_CUBE = "%s%s" %(Parent_Dir, Cube_Name)
        #FILE_CUBE = "%sMapFit/%s" %(Parent_Dir, Cube_Name)

        ## Get the mask name inkl its dir, if MASK is Ture
        if CalcParameters["MASK"] == True:
            Mask_Name = Mask_Names[i_set]
            FILE_MASK = "%s%s" %(Parent_Dir, Mask_Name)
            #FILE_MASK = "%sMapFit/%s" %(Parent_Dir, Mask_Name)

        else:
            FILE_MASK = "NoMask"
        
        #print('here')
        ## Calculate Feature Vectors
        #ts = time.time()
        FV_dset = Feature_Extractor(CalcParameters, FILE_CUBE, FILE_MASK, i_set+1)
        #te = time.time()
        #print(FV_dset)
        #print(FV_dset.describe())

        #print("It took %.3f s to calculate the FV." %(te-ts))

        ## Check if run was successful
        if isinstance(FV_dset, pd.DataFrame):

            ## Create a database out of the data set
            if data_set is None:
                FV_name = "%s_FV.pkl" %(".".join(Cube_Name.split(".")[:-1]))

                FV_dset.to_pickle("%s%s" %(CalcParameters["FVPath"], FV_name))

            #print("%s%s" %(CalcParameters["FVPath"], FV_name))

            ## Add FV name to database
            if data_set is None:
                data_pd.loc[i_set, "FV name"] = FV_name

        ## If the run failed, add the index to the list
        else:
            inval_cubes = np.append(inval_cubes, i_set)


    ## Remove all invalid data sets, if neccessary
    if list(inval_cubes):
        data_pd = data_pd.drop(index=inval_cubes)

    ## Update the database
    if data_set is None:
        data_pd.to_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]))

    ## Return whole CalcParameters or just the data set

    if data_set is not None:
        return data_set

    else:
        return
