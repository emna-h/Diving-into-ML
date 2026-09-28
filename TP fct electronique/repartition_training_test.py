import numpy as np
import scipy.io
from scipy.io import loadmat, savemat
import os
import platform
import sys


### Affichage visuel de démarcation (alternative à l'effacement console)
def clear_console(): 
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

# Appel immédiat au début du script
clear_console()    
    
np.random.seed(46) # 
# Chargement des matrices Minput et Moutput 
script_dir = os.path.dirname(os.path.abspath(__file__))

file_path_input = os.path.join(script_dir, 'Minput.npy')
Minput = np.load(file_path_input)

file_path_output = os.path.join(script_dir, 'Moutput.npy')
Moutput = np.load(file_path_output)

# FIN Chargement des matrices Minput et Moutput




# Mélange aléatoire des indices
total_cols = Minput.shape[1]
rand_indices = np.random.permutation(total_cols)

# Séparation 10% test / le reste% training
nb_test = round(0.10 * total_cols)
test_indices = rand_indices[:nb_test]
train_indices = rand_indices[nb_test:]

# Sélection des colonnes
Minput_test = Minput[:, test_indices]
Minput_training = Minput[:, train_indices]
Moutput_test = Moutput[:, test_indices]
Moutput_training = Moutput[:, train_indices]




# === Affichage d'une partie du Minput_training et Moutput_training ===
print("\nCaractéristiques de la matrice Minput_training :")
print(f"- Shape (lignes, colonnes) : {Minput_training.shape}")
print("→ Premières colonnes (extraits) :")
print(np.round(Minput_training[:, :5], 4))  

print("\nCaractéristiques de la matrice Moutput_training :")
print(f"- Shape (lignes, colonnes) : {Moutput_training.shape}")
print("→ Premières colonnes (extraits) :")
print(Moutput_training[:, :5].astype(int))  


# === Affichage d'une partie du Minput_test et Moutput_test ===
print("\nCaractéristiques de la matrice Minput_test :")
print(f"- Shape (lignes, colonnes) : {Minput_test.shape}")
print("→ Premières colonnes (extraits) :")
print(np.round(Minput_test[:, :5], 4))  

print("\nCaractéristiques de la matrice Moutput_test :")
print(f"- Shape (lignes, colonnes) : {Moutput_test.shape}")
print("→ Premières colonnes (extraits) :")
print(Moutput_test[:, :5].astype(int))  




# === Enregistrement de Minput et Moutput en format .mat dans le dossier contenant le code===
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Training ---
file_path = os.path.join(script_dir, 'Minput_training.mat')
scipy.io.savemat(file_path, {'Minput_training': Minput_training})

file_path = os.path.join(script_dir, 'Moutput_training.mat')
scipy.io.savemat(file_path, {'Moutput_training': Moutput_training})

# --- Test ---
file_path = os.path.join(script_dir, 'Minput_test.mat')
scipy.io.savemat(file_path, {'Minput_test': Minput_test})

file_path = os.path.join(script_dir, 'Moutput_test.mat')
scipy.io.savemat(file_path, {'Moutput_test': Moutput_test})




#### sauvegarde des matrices Minput et Moutput en format .npy dans le dossier contenant le code
# Récupérer le dossier du script courant
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construire le chemin complet vers le fichier à sauvegarder
file_path = os.path.join(script_dir, 'Minput_training.npy')

# Sauvegarder le fichier
np.save(file_path, Minput_training)

# Construire le chemin complet vers le fichier à sauvegarder
file_path = os.path.join(script_dir, 'Moutput_training.npy')

# Sauvegarder le fichier
np.save(file_path, Moutput_training)

# Construire le chemin complet vers le fichier à sauvegarder
file_path = os.path.join(script_dir, 'Minput_test.npy')

# Sauvegarder le fichier
np.save(file_path, Minput_test)

# Construire le chemin complet vers le fichier à sauvegarder
file_path = os.path.join(script_dir, 'Moutput_test.npy')

# Sauvegarder le fichier
np.save(file_path, Moutput_test)
#### FIN sauvegarde des matrices Minput et Moutput en format .npy dans le dossier contenant le code




