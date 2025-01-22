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
    # CSV-Datei in ein Pandas DataFrame laden und den Header explizit setzen
    squalo_df = pd.read_csv(csv_file_path, header=0)

    FILE_CUBE = squalo_df['Parent dir'] + squalo_df['Cube name']
    PREDICT = squalo_df['Parent dir'] + squalo_df['Predict name']
    Molecule = squalo_df['Molecule']
    distance = squalo_df['d']
    Temperature = squalo_df['T']
    
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

        # Calculate pixel size and get distance from CSV
        def calculate_pixel_size(Distanz):
            pixel_size_x = Distanz * (Cube_wcs.pixel_scale_matrix[0, 0] * 60 * 60)  # pc * Bogensekunden = au
            pixel_size_y = Distanz * (Cube_wcs.pixel_scale_matrix[1, 1] * 60 * 60)  # pc * Bogensekunden = au
            return pixel_size_x, pixel_size_y

        Distanz = distance[i] * 1000
        pixel_size_x, pixel_size_y = calculate_pixel_size(Distanz)
        pixel_fläche = np.abs(pixel_size_x * pixel_size_y)

        # Calculate radius
        Anzahl_Pixel = np.sum(Outflow_Moment0 > 0)
        Fläche_Outflow = Anzahl_Pixel * pixel_fläche
        Rlobe = np.sqrt(Fläche_Outflow / const.pi)

        # Check which molecule
        molecule = Molecule[i]

        ### funktion für partition function ###

        def boltzmann_distribution(energy, temperature):
                k = const.k  # Boltzmann-Konstante in J/K
                return np.exp(-energy / (k * temperature))

        ### Häufigkeiten setzen, und werte aus datenbank zuweisen ###

        if molecule == '12CO':
            Häufigkeit = 1.2 *10**4 #13Co Faktor 60 weniger als 12CO (wert aus paper)
            
            mean_molecular_weight = 14
            Einstein_coefiecient = 6.910e-07 ### richtig ###
            
            Energie_J2 = 11.534919938 * 1.98630e-23  # 1 cm^-1 = 1.98630e-23 J
            Energie_J1 = 3.845033413 * 1.98630e-23
            Z_2 = boltzmann_distribution(Energie_J2, np.nanmax(Outflow))
            Z_1 = boltzmann_distribution(Energie_J1, np.nanmax(Outflow))
            partition_function = Z_1 / Z_2
            
        if molecule == '13CO':
            Häufigkeit = 1/60 * 1.2 *10**4 #13Co Faktor 60 weniger als 12CO (wert aus paper)
            
            mean_molecular_weight = 14.5
            Einstein_coefiecient = 6.038e-07 ### richtig ###
            
            Energie_J2 = 11.0276302684 * 1.98630e-23  # 1 cm^-1 = 1.98630e-23 J
            Energie_J1 = 3.6759215030 * 1.98630e-23
            Z_2 = boltzmann_distribution(Energie_J2, np.nanmax(Outflow))
            Z_1 = boltzmann_distribution(Energie_J1, np.nanmax(Outflow))
            partition_function = Z_1 / Z_2
            
        if molecule == 'H2CO':
            Häufigkeit = 1 * 10**4
            
            mean_molecular_weight = 7.5
            Einstein_coefiecient = 2.818e-04 ### richtig ###
            
            Energie_J2 = 14.565500  * 1.98630e-23  # 1 cm^-1 = 1.98630e-23 J
            Energie_J1 = 7.286400 * 1.98630e-23
            Z_2 = boltzmann_distribution(Energie_J2, np.nanmax(Outflow))
            Z_1 = boltzmann_distribution(Energie_J1, np.nanmax(Outflow))
            partition_function = Z_1 / Z_2        

        ### Parameter bearbeiten ### 
        hbar = const.hbar
        k = const.k
        nu = Cube_header['RESTFRQ']
        u, unit, uncetrainty = const.physical_constants['atomic mass constant']
        mass_H = 1.00784 * u
        Fläche_pixel = pixel_size_x * pixel_size_y * astropyconst.au.value**2

        K = mean_molecular_weight * mass_H * np.abs(Fläche_pixel) * Häufigkeit * partition_function * ((8 * const.pi * const.k * Cube_header['RESTFRQ']**2) / (const.h * const.c**3 * Einstein_coefiecient))

        # Calculate channel width in m/s
        delta_v = (cube.spectral_axis[1] - cube.spectral_axis[0])

        if np.abs(delta_v.value) < 1000:
            delta_v = np.abs(delta_v) * 1000
        else:
            delta_v = np.abs(delta_v)

        # Calculate mass
        Outflow_Mass = np.nansum(Outflow_Moment0) * delta_v.value * K / astropyconst.M_sun.value

        # Plots
        data_bm, velax = bm.load_cube(FILE_CUBE[i])
        rms = bm.estimate_RMS(data=Outflow, N=5)

        #if np.abs(velax) > 1000:
        velax = velax / 1000
        #else:
            #velax = velax
        whole_Moment0, whole_Moment0_Error = bm.methods.collapse_zeroth(velax, data_bm, rms)
        Moment0, Moment0_Error = bm.methods.collapse_zeroth(velax, Outflow, rms)
        Moment1, Moment1_Error = bm.methods.collapse_first(velax, Outflow, rms)
        Moment2, Moment2_Error = bm.methods.collapse_second(velax, Outflow, rms)
        Max_Map, Max_Map_Error = bm.methods.collapse_eighth(velax, Outflow, rms)
 

        Vmax = np.nanmax(Moment1)
        Vmin = np.nanmin(Moment1)

        V_outflow = Vmax - Vmin

        Daten = np.array([np.abs(Outflow_Mass), np.abs(V_outflow) * 1000, Rlobe])

        objects = [squalo_df['Cube name'][i]]
        parameter = ['Mass (M_sun)', 'Vmax (m/s)', 'Rlobe (AU)']
        calculation_df = pd.DataFrame(data=[Daten], index=objects, columns=parameter)

        calculation_df['Momentum (M_sun*km/s)'] = calculation_df['Mass (M_sun)'] * calculation_df['Vmax (m/s)'] / 1000
        calculation_df['Time (yr)'] = abs(((calculation_df['Rlobe (AU)'] * astropyconst.au.value) / (calculation_df['Vmax (m/s)'])) / (60 * 60 * 24 * 365))
        calculation_df['Mass outflow rate (M_sun/yr)'] = calculation_df['Mass (M_sun)'] / calculation_df['Time (yr)']
        calculation_df['Outflow Force (M_sun*km/s*yr^-1)'] = calculation_df['Mass (M_sun)'] * astropyconst.M_sun.value * (calculation_df['Vmax (m/s)'])**2 / (calculation_df['Rlobe (AU)'] * astropyconst.au.value) * ((60 * 60 * 24 * 365) / (1000 * astropyconst.M_sun.value))
        calculation_df['E_kin (M_sun*(km/s)^2)'] = 1 / 2 * calculation_df['Mass (M_sun)'] * (calculation_df['Vmax (m/s)'] / 1000)**2
        calculation_df['Luminosity (L_sun)'] = ((calculation_df['E_kin (M_sun*(km/s)^2)'] * astropyconst.M_sun.value * (1000**2)) / (calculation_df['Time (yr)'] * 60 * 60 * 24 * 365)) / astropyconst.L_sun.value

        speicherpfad_tabelle = os.path.join(squalo_df['Parent dir'][i], 'Calculations')
        if not os.path.exists(speicherpfad_tabelle):
            os.makedirs(speicherpfad_tabelle)
        dateiname = f"{squalo_df['Cube name'][i]}_calculations.csv"
        calculation_df.to_csv(os.path.join(speicherpfad_tabelle, dateiname))


        table = calculation_df.iloc[:, :3].copy()

        table = table.rename(columns={table.columns[0]: 'Mass (M☉) :'})
        table = table.rename(columns={table.columns[1]: 'Velocity (km/s) :'})
        table = table.rename(columns={table.columns[2]: 'Radius (AU) :'})

        table['Mass (M☉) :'] = round(table['Mass (M☉) :'],2)
        table['Velocity (km/s) :'] = round(table['Velocity (km/s) :']/1000,2)
        table['Radius (AU) :'] = round(table['Radius (AU) :']).astype(int)

        # Wandle die Tabelle in das gewünschte Format um und entferne die Spalte "Property"
        table = table.stack().reset_index().rename(columns={'level_0': 'Property', 'level_1': 'Index', 0: 'Value'})

        # Füge Zeilenumbrüche und Leerzeichen zwischen den Einträgen hinzu
        table_text = ''
        for index, row in table.iterrows():
            table_text += f"{row['Index']} {row['Value']}\n\n"
            if index % 3 == 2 and index != len(table)-1:  # Füge zusätzlichen Zeilenumbruch ein, außer beim letzten Eintrag
                table_text += '\n'

        # Entferne Dezimalstellen für die Spalte "Radius"
        table_text = table_text.replace('.0', '')        



        # Erstellen des Subplots
        fig, axs = plt.subplots(3, 2, figsize=(10, 12), subplot_kw={'projection': wcs_info})
        fig.suptitle(FILE_CUBE, fontsize=20)

        # Plot whole_Moment0 (oben links)
        im = axs[0, 0].imshow(whole_Moment0)
        axs[0, 0].set_title('Whole Moment 0')
        cb = fig.colorbar(im, ax=axs[0, 0], fraction=0.046, pad=0.04)  # Anpassung der Farbskala
        cb.set_label('K')  # Farbskalenbeschriftung
        axs[0, 0].set_xlabel('RA')
        axs[0, 0].set_ylabel('DEC')

        # Erstelle ein neues Subplot für die Tabelle
        axs[0, 1].axis('off')  # Schalte Achsen für die Tabelle aus
        axs[0, 1].text(0.5, 0.5, table_text, fontsize=16, va='center', ha='center', multialignment='left')

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
        cb.set_label('K')  # Farbskalenbeschriftung
        axs[1, 1].set_xlabel('RA')
        axs[1, 1].set_ylabel('DEC')

        # Plot Moment 1 (Mitte rechts)
        im = axs[2, 0].imshow(Moment1, cmap='seismic')  # cmap='seismic' für seismisches Format
        axs[2, 0].set_title('Moment 1')
        cb = fig.colorbar(im, ax=axs[2, 0], fraction=0.046, pad=0.04)  # Anpassung der Farbskala
        cb.set_label('K * km/s')  # Farbskalenbeschriftung
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
        
outflow_analysis('Squalo_Info.csv')
