import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import SpectralCube as sc
from astropy.stats import sigma_clip

def volume_analysis(csv_file_path):
    # CSV-Datei in ein Pandas DataFrame laden und den Header explizit setzen
    df = pd.read_csv(csv_file_path, header=0)

    FILE_CUBE = df['Parent dir'] + df['Cube name']
    PREDICT = df['Parent dir'] + df['Predict name']

    # Liste zum Speichern der Ergebnisse initialisieren
    results = []

    for i in range(len(FILE_CUBE)):
    
        cube = sc.read(FILE_CUBE[i]).to("K")
        data = cube.hdulist[0].data
        cube_predict = sc.read(PREDICT[i])
        predict = cube_predict.hdulist[0].data

        Cube_header = cube.header
        Cube_wcs = WCS(Cube_header)

        def remove_nan_channels(data):
            nan_indices = np.all(np.isnan(data), axis=(1, 2))
            non_nan_data = data[~nan_indices]
            print('Channels von', data.shape[0], 'auf', non_nan_data.shape[0], 'reduziert')
            return non_nan_data

        if np.any(np.all(np.isnan(data), axis=(1, 2))):
            data = remove_nan_channels(data)

        # Berechnung des Outflow
        Outflow = data * predict

        # Voxel zählen, deren Werte über 0 liegen
        positive_voxels = np.sum(predict == 1)

        # Auflösung und räumliche Breite aus dem Header extrahieren
        delta_v = Cube_header['CDELT3']
        if np.abs(delta_v) > 100:
            delta_v = np.abs(delta_v)/1000
        else:
            delta_v = np.abs(delta_v)
        resolution = delta_v  # Kanalbreite (Geschwindigkeit)
        
        spatial_width = np.abs(Cube_header['CDELT1']) * np.abs(Cube_header['CDELT2'])  # Fläche pro Pixel (räumliche Dimensionen)
        
        # Volumen berechnen
        voxel_volume = resolution * spatial_width
        total_volume = positive_voxels * voxel_volume

        # Sigma-Clip durchführen
        data_sc = sigma_clip(data, sigma=3, maxiters=None, cenfunc='mean')

        # Daten, die verworfen wurden (True in der Maske)
        rejected_data = data[data_sc.mask]

        # Nur positive Werte in rejected_data beibehalten
        rejected_data = rejected_data[rejected_data > 0]

        # Mittelwert der verworfenen Daten berechnen
        mean_rejected_data = np.nanmean(rejected_data)

        # Standardabweichung der gesclipten Daten berechnen
        scstd = np.nanstd(data_sc)

        # Signal-Rausch-Verhältnis berechnen
        signal_to_noise = mean_rejected_data / scstd

        # Summe des Outflows entlang der Achse 0 berechnen
        sum_outflow = np.nansum(Outflow, axis=0)

        # Positive Pixel und Fläche berechnen
        positive_pixel = np.sum(sum_outflow > 0)
        area = positive_pixel * spatial_width  # Fläche in Grad-Einheit (deg^2)

        # Summenintensität berechnen
        sum_intensity = np.sum(sum_outflow)

        # Durchschnittliches Signal im Outflow
        mean_signal_outflow = np.nanmean(Outflow[Outflow > 0])

        # Filtere positive Daten
        positive_mask = data > 0
        positive_data = data[positive_mask]
        
        # Gleiche Maske auf predict anwenden und dann nach predict == 0 filtern
        predict_positive = predict[positive_mask]
        mean_signal_no_predict = np.nanmean(positive_data[predict_positive == 0])

        # Anzahl der Datenpunkte in data, die größer als 3x scstd sind
        num_points_above_3scstd = np.nansum(data > (3 * scstd))

        # Gesamtanzahl der Datenpunkte in data
        total_data_points = data.size

        # Ergebnisse speichern
        results.append([
            df['Cube name'][i], positive_voxels, total_volume, 
            resolution, spatial_width, signal_to_noise, 
            positive_pixel, area, sum_intensity, 
            mean_rejected_data, scstd, mean_signal_outflow, 
            mean_signal_no_predict, num_points_above_3scstd,
            total_data_points
        ])

    # Ergebnisse in ein Pandas DataFrame umwandeln
    results_df = pd.DataFrame(results, columns=[
        'File_Cube', 'Positive_Voxels', 'Total_Volume', 
        'Resolution', 'Spatial_Width', 'Signal_to_Noise', 
        'Positive_Pixel', 'Area', 'Sum_Intensity', 
        'Mean_Rejected_Data', 'SCSTD', 'Mean_Signal_Outflow', 
        'Mean_Signal_No_Predict', 'Num_Points_Above_3SCSTD',
        'Total_Data_Points'
    ])

    # DataFrame als CSV-Datei speichern
    results_df.to_csv('Volume_Analysis_Results.csv', index=False)

# Aufruf der Funktion mit dem Pfad zur CSV-Datei
volume_analysis('Volume_analysis.csv')
