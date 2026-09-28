import torch 
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.io import savemat

def clear_console(): 
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")
clear_console()    

# === Paramètres modifiables : ces paramètres doivent être similaires à ceux utilisés en dataset.py ===
###########################
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
max_rings = 5
###########################






# === Chargement des données ===
script_dir = os.path.dirname(os.path.abspath(__file__))
Minput_FFNN_test = np.load(os.path.join(script_dir, 'Minput_test.npy')).T
Moutput_FFNN_test = np.load(os.path.join(script_dir, 'Moutput_test.npy')).T
# En apprentissage machine, on veut généralement que :
# Chaque ligne correspond à un échantillon
# Chaque colonne correspond à une spécification
# on fait .T (transposée) pour remettre les exemples (échantillons) en lignes.

Minput_complete = np.load(os.path.join(script_dir, 'Minput.npy'))

norm_params = np.load(os.path.join(script_dir, 'normalization_params.npz'))
Minput_min = norm_params['Minput_min']
Minput_max = norm_params['Minput_max']
Moutput_min = norm_params['Moutput_min']
Moutput_max = norm_params['Moutput_max']

Minput_norm_test = (Minput_FFNN_test - Minput_min) / (Minput_max - Minput_min + 1e-8)
# Normalisation de Minput_norm_test

# === Sélection de l’exemple ===
indice_test = 10
x_test = torch.tensor(Minput_norm_test[indice_test], dtype=torch.float32)

# === Définition du modèle === on doit définir le même modèle utilisé dans la phase de training
class FFNN(nn.Module): #On crée une classe qui hérite de nn.Module (la classe de base pour tous les modèles PyTorch).
    def __init__(self, input_size, hidden1, hidden2, hidden3, output_size, dropout_rate=0.000): # __init__ initialise les couches du réseau.
        super(FFNN, self).__init__()#super(FFNN, self).__init__() appelle le constructeur de nn.Module pour activer toute la mécanique PyTorch (gestion des poids, gradients, etc.).
        self.fc1 = nn.Linear(input_size, hidden1)#fc1 : couche fully connected (dense) qui prend input_size entrées et produit hidden1 sorties.
        self.relu1 = nn.ReLU()#relu1 : fonction d’activation ReLU appliquée après fc1 pour introduire de la non-linéarité.
        self.dropout1 = nn.Dropout(p=dropout_rate)  # ajout dropout après fc1. p est la probabilité de mettre un neurone à zéro.Par défaut, souvent p = 0.5.Exemple : si p=0.3, alors 30% des neurones sont éteints aléatoirement à chaque passage.
        
        self.fc2 = nn.Linear(hidden1, hidden2)#fc2 : deuxième couche dense, de hidden1 → hidden2.
        self.relu2 = nn.ReLU()#relu2 : activation ReLU après fc2.
        self.dropout2 = nn.Dropout(p=dropout_rate)  # ajout dropout après fc2
        
        self.fc3 = nn.Linear(hidden2, hidden3)#fc3 : troisième couche dense, de hidden2 → hidden3.
        self.relu3 = nn.ReLU()#relu3 : activation ReLU après fc3.
        self.dropout3 = nn.Dropout(p=dropout_rate)  # ajout dropout après fc3
        
        self.fc4 = nn.Linear(hidden3, output_size)#fc4 : couche finale dense, qui projette hidden3 → output_size. Pas de fonction d’activation ici → sortie brute

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)       # applique dropout
        
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)       # applique dropout
        
        x = self.relu3(self.fc3(x))
        x = self.dropout3(x)       # applique dropout
        
        return self.fc4(x)

# === Chargement du modèle entraîné ===
input_size = Minput_norm_test.shape[1]
output_size = Moutput_FFNN_test.shape[1]

model = FFNN(input_size=input_size, hidden1=14, hidden2=20, hidden3=8, output_size=output_size, dropout_rate=0.000)
#Ici on crée une instance du réseau de neurones FFNN.
#input_size = nombre de neurones en entrée (taille des données d’entrée).
#10, 10, 10 = tailles des trois couches cachées (3 couches de 10 neurones chacune).
#output_size = nombre de neurones en sortie (taille de ce que tu veux prédire).

model.load_state_dict(torch.load(os.path.join(script_dir, 'ffnn_model.pth')))
# Ici on charge les poids appris par le modèle.
# torch.load(...) = lit le fichier contenant les paramètres sauvegardés (ici ffnn_model.pth).
# model.load_state_dict(...) = injecte ces paramètres dans le modèle FFNN que tu viens d’instancier.
# Attention : tu dois recréer ton modèle avec la même architecture que lors de l’entraînement, sinon le chargement échoue.

model.eval()
# Cette commande met le modèle en mode évaluation :
# Certaines couches comme Dropout ou BatchNorm se comportent différemment entre entraînement et test.
# En mode eval(), elles sont figées pour donner des résultats cohérents lors de prédiction.


with torch.no_grad(): # Cela désactive le calcul du gradient. En phase de test / évaluation, on n’a pas besoin de calculer les gradients
    y_pred_norm = model(x_test) # On passe les données de test (x_test) dans le modèle (model).
y_pred = y_pred_norm.numpy() * (Moutput_max - Moutput_min + 1e-8) + Moutput_min 
# Ici, on retransforme les données normalisées en vraies valeurs physiques (dénormalisation). 
# Après normalisation : Xnorm​= (​X−Xmin​)/(Xmax​−Xmin). Après dénormalisation :​ X=Xnorm​⋅(Xmax​−Xmin​)+Xmin​
# y_pred_norm est un Tensor PyTorch, le résultat est un torch.Tensor, car tous les calculs
# dans PyTorch se font avec des tenseurs.  .numpy() convertit le Tensor en array NumPy

architecture_predicted = np.round(y_pred).astype(int)#
architecture_reference = Moutput_FFNN_test[indice_test].astype(int)

print("Architecture référence  :", architecture_reference)
print("Architecture prédite    :", architecture_predicted)




# === Constantes ===

k = 2 * np.pi / lambda_
theta0deg = Minput_FFNN_test.T[3, indice_test]
theta0 = np.deg2rad(theta0deg)
phi0 = 0
phi = 0
theta = np.linspace(0, 2 * np.pi, 1000)
theta_deg = np.rad2deg(theta)


# === Fonction : calcul du diagramme et des métriques ===
def calcul_AF_performance_metrics(elements_per_ring):
    rings = len(elements_per_ring)
    radii = r0 + delta_r * np.arange(max_rings)
    
    AF_az = np.zeros_like(theta, dtype=complex)
    
    for ring in range(rings):
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
    halfPower = maxVal_dB - 3#
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
        main_lobe_gain = maxVal_non_norm 
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
        
    return AF_dB_az, HPBW, maxVal_non_norm, true_SSL_gain








# === Calcul des performances ===
AF_ref_dB, HPBW_ref, gain_main_ref, SSL_ref = calcul_AF_performance_metrics(architecture_reference)
AF_pred_dB, HPBW_pred, gain_main_pred, SSL_pred = calcul_AF_performance_metrics(architecture_predicted)

# === Tracé polaire ===
plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)
ax.plot(theta, AF_ref_dB, label='Référence', linewidth=2)
ax.plot(theta, AF_pred_dB, label='Prédiction', linewidth=2, linestyle='--')
ax.set_title("Diagramme de rayonnement (polaire)", fontsize=13)
ax.set_rlim([-40, 0])
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# === Tracé cartésien ===
plt.figure(figsize=(10, 5))
plt.plot(theta_deg, AF_ref_dB, label='Référence', linewidth=2)
plt.plot(theta_deg, AF_pred_dB, label='Prédiction', linewidth=2, linestyle='--')
plt.xlabel("Azimut (°)")
plt.ylabel("Gain (dB)")
plt.title("Diagramme de rayonnement - Azimut")
plt.xlim([0, 360])
plt.ylim([-40, 0])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# === Affichage de theta0deg ===
print(f"\ntheta0deg                 : {theta0deg:.2f}°")

# === Affichage des performances ===
print(f"\n--- Performances Référence ---")
print(f"Gain du lobe principal : {gain_main_ref:.2f} dB")
print(f"True Side Lobe Gain   : {SSL_ref:.2f} dB")
print(f"HPBW                  : {HPBW_ref:.2f}°")


print(f"\n--- Performances Prédiction ---")
print(f"Gain du lobe principal : {gain_main_pred:.2f} dB")
print(f"True Side Lobe Gain   : {SSL_pred:.2f} dB")
print(f"HPBW                  : {HPBW_pred:.2f}°")




# === Calcul de l'erreur MAE entre l'architecture prédite et l'architecture de référence ===
mae_HPBW = abs(HPBW_pred - HPBW_ref)
mae_main_lobe_gain = abs(gain_main_pred - gain_main_ref)
mae_true_SSL_gain = abs(SSL_pred - SSL_ref)

print("\n=== Erreurs absolues (MAE) entre prédiction et référence ===")
print(f"Erreur Gain Lobe Principal : {mae_main_lobe_gain:.2f} dB")
print(f"Erreur True SSL Gain      : {mae_true_SSL_gain:.2f} dB")
print(f"Erreur HPBW               : {mae_HPBW:.2f} degrés")



# --- Extraction des valeurs maximales ---
max_main_lobe_gain = np.max(Minput_complete[0, :])
max_true_SSL_gain  = np.max(np.abs(Minput_complete[1, :]))  
max_HPBW          = np.max(Minput_complete[2, :])

# # --- Affichage ---
# print("\n=== valeurs max ===")
# print(f"Valeur max du main_lobe_gain : {max_main_lobe_gain:.2f} dB")
# print(f"Valeur max du true_SSL_gain  : {max_true_SSL_gain:.2f} dB (en valeur absolue)")
# print(f"Valeur max du HPBW           : {max_HPBW:.2f} degrés")



# === Coefficients de pondération ===
w1 = 0.33
w2 = 0.33
w3 = 0.34

# === Calcul des erreurs relatives (en %) ===
err_rel_main_lobe_gain = 100*(abs(gain_main_pred - gain_main_ref) / (abs(max_main_lobe_gain) + 1e-8))
err_rel_true_SSL_gain  = 100*(abs(SSL_pred - SSL_ref) / (abs(max_true_SSL_gain) + 1e-8))
err_rel_HPBW           = 100*(abs(HPBW_pred - HPBW_ref) / (180 + 1e-8))

# === Calcul de l'erreur de prediction globale pondérée ===
error_pred_global = (    w1 * err_rel_main_lobe_gain +    w2 * err_rel_true_SSL_gain  +    w3 * err_rel_HPBW  )     


print("\n=== Erreurs relatives de prédiction (%) ===")
print(f"Erreur relative de prédiction Gain Lobe Principal : {err_rel_main_lobe_gain:.2f} %")
print(f"Erreur relative de prédiction True SSL Gain      : {err_rel_true_SSL_gain:.2f} %")
print(f"Erreur relative de prédiction HPBW               : {err_rel_HPBW:.2f} %")

print(f"\nErreur globale de prédiction pondérée : {error_pred_global:.2f} %")



