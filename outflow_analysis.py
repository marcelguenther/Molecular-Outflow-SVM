import os
import matplotlib.pyplot as plt
import numpy as np
import astropy
import math
from astropy.wcs import WCS
from astropy.wcs import wcs
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy import constants as astropyconst
from astropy import units as u
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
import spectral_cube
from spectral_cube import SpectralCube as sc
import scipy
from scipy import constants as const
import pandas as pd
import bettermoments as bm

def outflow_analysis(csv_file_path):
    # load CSV-Data in Pandas
    squalo_df = pd.read_csv(csv_file_path, header=0)

    FILE_CUBE = squalo_df['Parent dir'] + squalo_df['Cube name']
    PREDICT = squalo_df['Parent dir'] + squalo_df['Predict name']
    
    for i in range(len(FILE_CUBE)):
        cube = sc.read(FILE_CUBE[i]).to("K")
        data = cube.hdulist[0].data
        cube_predict = sc.read(PREDICT[i])
        predict = cube_predict.hdulist[0].data

        Cube_header = cube.header
        Cube_wcs = WCS(Cube_header)

        wcs_info = wcs.WCS(Cube_header)
        wcs_info = wcs_info.dropaxis(2)

        def remove_nan_channels(data):
            # Überprüfen, ob die Daten NaN-Werte enthalten
            nan_indices = np.all(np.isnan(data), axis=(1, 2))  # Indizes der Kanäle mit NaN-Werten finden
            non_nan_data = data[~nan_indices]  # Kanäle ohne NaN-Werte auswählen
            print('Channels von', data.shape[0] ,'auf', non_nan_data.shape[0],'reduziert')
            return non_nan_data

        if np.any(np.all(np.isnan(data), axis=(1, 2))):
            data = remove_nan_channels(data)

        Outflow = data * predict
        Outflow_Moment0 = np.nansum(Outflow, axis=0)

        # Plots
        data_bm, velax = bm.load_cube(FILE_CUBE[i])
        rms = bm.estimate_RMS(data=Outflow, N=5)

        whole_Moment0, whole_Moment0_Error = bm.methods.collapse_zeroth(velax, data_bm, rms)
        Moment0, Moment0_Error = bm.methods.collapse_zeroth(velax, Outflow, rms)
        Moment1, Moment1_Error = bm.methods.collapse_first(velax, Outflow, rms)
        Moment2, Moment2_Error = bm.methods.collapse_second(velax, Outflow, rms)
        Max_Map, Max_Map_Error = bm.methods.collapse_eighth(velax, Outflow, rms)
 

        # Erstellen des Subplots
        fig, axs = plt.subplots(3, 2, figsize=(10, 12), subplot_kw={'projection': wcs_info})
        fig.suptitle(squalo_df['Cube name'][i], fontsize=16)

        # Plot whole_Moment0 (oben links)
        im = axs[0, 0].imshow(whole_Moment0)
        axs[0, 0].set_title('Whole Moment 0')
        cb = fig.colorbar(im, ax=axs[0, 0], fraction=0.046, pad=0.04)  # Anpassung der Farbskala
        cb.set_label('K')  # Farbskalenbeschriftung
        axs[0, 0].set_xlabel('RA')
        axs[0, 0].set_ylabel('DEC')

        # Erstelle ein neues Subplot für die Tabelle
        axs[0, 1].axis('off')  # Schalte Achsen für die Tabelle aus

        # Plot Max Map (oben rechts)
        im = axs[1, 0].imshow(Max_Map)
        axs[1, 0].set_title('Max Map')
        cb = fig.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)  # Anpassung der Farbskala
        cb.set_label('K')  # Farbskalenbeschriftung
        axs[1, 0].set_xlabel('RA')
        axs[1, 0].set_ylabel('DEC')

        # Plot Moment 0 (Mitte links)
        im = axs[1, 1].imshow(Moment0)
        axs[1, 1].set_title('Moment 0')
        cb = fig.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)  # Anpassung der Farbskala
        cb.set_label('K * km/s')  # Farbskalenbeschriftung
        axs[1, 1].set_xlabel('RA')
        axs[1, 1].set_ylabel('DEC')

        # Plot Moment 1 (Mitte rechts)
        im = axs[2, 0].imshow(Moment1, cmap='seismic')  # cmap='seismic' für seismisches Format
        axs[2, 0].set_title('Moment 1')
        cb = fig.colorbar(im, ax=axs[2, 0], fraction=0.046, pad=0.04)  # Anpassung der Farbskala
        cb.set_label('km/s')  # Farbskalenbeschriftung
        axs[2, 0].set_xlabel('RA')
        axs[2, 0].set_ylabel('DEC')

        # Plot Moment 2 (unten links)
        im = axs[2, 1].imshow(Moment2)
        axs[2, 1].set_title('Moment 2')
        cb = fig.colorbar(im, ax=axs[2, 1], fraction=0.046, pad=0.04)  # Anpassung der Farbskala
        cb.set_label('km/s')  # Farbskalenbeschriftung
        axs[2, 1].set_xlabel('RA')
        axs[2, 1].set_ylabel('DEC')

        # Anpassen der Layouts
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)  # Anpassung des Abstands

        speicherpfad = os.path.join(squalo_df['Parent dir'].iloc[i], 'Plots')
        if not os.path.exists(speicherpfad):
            os.makedirs(speicherpfad)

        dateiname_prefix = squalo_df['Cube name'].iloc[i]
        plt.savefig(os.path.join(speicherpfad, f'{dateiname_prefix}_plots.png'))
        
        plt.show()
        
outflow_analysis('Cubes-Info.csv')
