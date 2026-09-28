import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import scipy.io
import os
import platform
import sys



### Affichage visuel de démarcation (alternative à l'effacement console)
def clear_console(): 
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

# Appel immédiat au début du script
clear_console()



# === Paramètres modifiables ===
nb_samples = 1000
max_rings = 5
max_elements = 10
theta0_max_deg = 180
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
k = 2 * np.pi / lambda_
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
np.random.seed(46) 
# === Initialisation des matrices ===

Moutput = np.zeros((max_rings, nb_samples))  # Architecture
Minput = np.zeros((4, nb_samples))           # [MainLobe, SSL, HPBW, theta0]

# === Boucle principale ===
for sample_idx in range(nb_samples):
    # --- Architecture complètement aléatoire ---
    while True:
        elements_per_ring = np.random.randint(0, max_elements + 1, size=max_rings)
        if np.sum(elements_per_ring > 0) > 0:  # S'assurer qu'au moins un ring est actif
            break
    Moutput[:, sample_idx] = elements_per_ring

    # --- Theta0 aléatoire ---
    theta0deg = np.random.uniform(0, theta0_max_deg)
    theta0 = np.deg2rad(theta0deg)
    phi0 = 0

    # --- Rayons des anneaux ---
    radii = r0 + delta_r * np.arange(max_rings)

    # --- Diagramme 2D ---
    theta = np.linspace(0, 2 * np.pi, 1000)
    phi = 0
    AF_az = np.zeros_like(theta, dtype=complex)

    for ring in range(max_rings):
        a = radii[ring]
        N = elements_per_ring[ring]
        if N == 0:
            continue
        phi_n = 2 * np.pi * np.arange(N) / N
        for n in range(N):
            phase = k * a * (np.sin(theta) * np.cos(phi - phi_n[n]) -
                             np.sin(theta0) * np.cos(phi0 - phi_n[n]))
            AF_az += np.exp(1j * phase)

    AF_norm_az = np.abs(AF_az) / (np.max(np.abs(AF_az)) + np.finfo(float).eps)
    AF_dB_az = 20 * np.log10(AF_norm_az + np.finfo(float).eps)
    AF_dB_az[AF_dB_az < -40] = -40
    theta_deg = np.rad2deg(theta)
    
    #calcul du gain max non normalisé du lobe principale
    AF_abs_az = np.abs(AF_az)
    maxVal = np.max(AF_abs_az)
    maxVal_non_norm = 20 * np.log10(maxVal + np.finfo(float).eps)
    
    # --- HPBW ---
    maxVal_dB = np.max(AF_dB_az)
    maxIdx = np.argmax(AF_dB_az)
    halfPower = maxVal_dB - 3
    AF_dB_ext = np.concatenate((AF_dB_az, AF_dB_az, AF_dB_az))# 
    theta_deg_ext = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))# 
    maxIdx_ext = maxIdx + len(theta_deg)#
    
    leftIdx_ext = np.where(AF_dB_ext[:maxIdx_ext] <= halfPower)[0][-1] if np.any(
        AF_dB_ext[:maxIdx_ext] <= halfPower) else None #
    
    rightIdx_ext = np.where(AF_dB_ext[maxIdx_ext:] <= halfPower)[0]
    
        
    rightIdx_ext = rightIdx_ext[0] + maxIdx_ext if len(rightIdx_ext) > 0 else None
    
    if leftIdx_ext is None or rightIdx_ext is None:
        HPBW = 180
    else:
        HPBW = theta_deg_ext[rightIdx_ext] - theta_deg_ext[leftIdx_ext]

    # --- Gain lobe principal & SSL ---
    responseLin = AF_norm_az#
    peaks, _ = find_peaks(responseLin, distance=5)
    
    pk = responseLin[peaks]#
    if len(pk) == 0:#
        main_lobe_gain = maxVal_non_norm # 
        true_SSL_gain = 0# 
    else: 
        sorted_idx = np.argsort(pk)[::-1]#
        sorted_pk = pk[sorted_idx]#
        sorted_angles = theta_deg[peaks][sorted_idx]#

        threshold_dB = 1#
        main_lobes_idx = np.where(
            20 * np.log10(sorted_pk + np.finfo(float).eps) >=
            20 * np.log10(sorted_pk[0] + np.finfo(float).eps) - threshold_dB)[0]#
        side_lobe_idx = np.setdiff1d(np.arange(len(sorted_pk)), main_lobes_idx)#
        
        main_lobe_gain = maxVal_non_norm
        
        true_SSL_gain = 20 * np.log10(sorted_pk[side_lobe_idx[0]] + np.finfo(float).eps) if len(
            side_lobe_idx) > 0 else 0
        
    
    # --- Stockage ---
    Minput[0, sample_idx] = main_lobe_gain
    Minput[1, sample_idx] = true_SSL_gain
    Minput[2, sample_idx] = HPBW
    Minput[3, sample_idx] = theta0deg
    
    
# === Affichage d'une partie du Minput et Moutput ===
print("\nCaractéristiques de la matrice Minput :")
print(f"- Shape (lignes, colonnes) : {Minput.shape}")
print("→ Premières colonnes (extraits) :")
print(np.round(Minput[:, :5], 4))  

print("\nCaractéristiques de la matrice Moutput :")
print(f"- Shape (lignes, colonnes) : {Moutput.shape}")
print("→ Premières colonnes (extraits) :")
print(Moutput[:, :5].astype(int))  



# === Enregistrement de Minput et Moutput en format .mat dans le dossier contenant le code===
script_dir = os.path.dirname(os.path.abspath(__file__))

# Chemin du fichier .mat à enregistrer
fichier_path_Minput_mat = os.path.join(script_dir, 'Minput.mat')

# Enregistrement de la matrice Minput (anciennement Moutput) au format .mat
scipy.io.savemat(fichier_path_Minput_mat, {'Minput': Minput})

# Chemin du fichier .mat à enregistrer
fichier_path_Moutput_mat = os.path.join(script_dir, 'Moutput.mat')

# Enregistrement de la matrice Moutput au format .mat
scipy.io.savemat(fichier_path_Moutput_mat, {'Moutput': Moutput})

    
    
    
# === Enregistrement de Minput et Moutput en format .npy dans le dossier contenant le code===
script_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(script_dir, 'Minput.npy')
np.save(file_path, Minput)

file_path = os.path.join(script_dir, 'Moutput.npy')
np.save(file_path, Moutput)


print("✅ Minput.npy et Moutput.npy ont été enregistrés dans le dossier courant.")