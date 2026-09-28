import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import platform
import os
import matplotlib.pyplot as plt
import time



### Affichage visuel de démarcation (alternative à l'effacement console)
def clear_console(): 
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

# Appel immédiat au début du script
clear_console()    
    

np.random.seed(46) 

# === Chargement des données du dossier qui contient le script===
script_dir = os.path.dirname(os.path.abspath(__file__))

Minput_training = np.load(os.path.join(script_dir, 'Minput_training.npy'))
Moutput_training = np.load(os.path.join(script_dir, 'Moutput_training.npy'))

# Transposer si nécessaire
if Minput_training.shape[0] < Minput_training.shape[1]:
    Minput_training = Minput_training.T
if Moutput_training.shape[0] < Moutput_training.shape[1]:
    Moutput_training = Moutput_training.T



# === Normalisation (Min-Max) ===
Minput_min = Minput_training.min(axis=0)
Minput_max = Minput_training.max(axis=0)
Moutput_min = Moutput_training.min(axis=0)
Moutput_max = Moutput_training.max(axis=0)

Minput_norm = (Minput_training - Minput_min) / (Minput_max - Minput_min + 1e-8)
Moutput_norm = (Moutput_training - Moutput_min) / (Moutput_max - Moutput_min + 1e-8)




# Conversion en tenseurs PyTorch
X = torch.tensor(Minput_norm, dtype=torch.float32)  # input du modèle
Y = torch.tensor(Moutput_norm, dtype=torch.float32)   # output à prédire



# === DataLoader ===
dataset = TensorDataset(X, Y)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# === Définition du modèle FFNN ===
class FFNN(nn.Module): #On crée une classe qui hérite de nn.Module 
    def __init__(self, input_size, hidden1, hidden2, hidden3, output_size, dropout_rate=0.000): # __init__ initialise les couches du réseau.
        super(FFNN, self).__init__()#super(FFNN, self).__init__() appelle le constructeur de nn.Module pour activer toute la mécanique PyTorch (gestion des poids, gradients, etc.).
        self.fc1 = nn.Linear(input_size, hidden1)#fc1 : couche fully connected (dense) qui prend input_size entrées et produit hidden1 sorties.
        self.relu1 = nn.ReLU()#relu1 : fonction d’activation ReLU appliquée après fc1 pour introduire de la non-linéarité.
        self.dropout1 = nn.Dropout(p=dropout_rate)  # ajout dropout après fc1. p est la probabilité de mettre un neurone à zéro.
        
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


#Donc ce réseau est un FFNN à 3 couches cachées avec activations ReLU et une couche de sortie linéaire.

# === Dimensions ===
input_size = X.shape[1]
output_size = Y.shape[1]

# === Modèle, perte et optimiseur ===
model = FFNN(input_size=input_size, hidden1=14, hidden2=20, hidden3=8, output_size=output_size, dropout_rate=0.000)
# Ici tu crées une instance du réseau de neurones FFNN (Feed-Forward Neural Network).
# input_size : dimension d’entrée (nombre de caractéristiques dans Minput_norm).
# hidden1, hidden2, hidden3 = 10 : chaque couche cachée a 10 neurones.
# output_size : dimension de sortie (nombre de valeurs que tu veux prédire, venant de Moutput_norm).


criterion = nn.MSELoss()
# nn.MSELoss() = Mean Squared Error Loss (erreur quadratique moyenne).
# Elle mesure la différence entre les sorties du modèle (prédictions) et les vraies valeurs (cibles).
#criterion = fonction qui mesure l’erreur entre prédiction et réalité (MSE).


optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# torch.optim.Adam est un algorithme d’optimisation.
# Il met à jour les poids et biais du modèle pour minimiser la perte.
# model.parameters() = tous les paramètres entraînables du réseau (poids et biais des couches).
# lr=0.001 = learning rate
#optimizer = méthode (Adam) pour ajuster les paramètres du modèle afin de réduire la perte.

# === Entraînement ===
num_epochs = 100
loss_list = []
# num_epochs : nombre de fois où le modèle parcourt tout le dataset. Ici, 100 passages.
# loss_list : liste pour stocker l’évolution de la perte à chaque époque (utile pour tracer la courbe d’apprentissage).

start_time = time.time()  # début du training
for epoch in range(num_epochs):
    total_loss = 0.0 #total_loss initialise la somme des pertes pour cette époque.
    for batch_X, batch_Y in dataloader:#Boucle sur chaque mini-batch (ici taille = 64) du DataLoader.
        optimizer.zero_grad()#On réinitialise les gradients à zéro avant chaque étape de mise à jour. Sinon, PyTorch accumule les gradients d’une itération à l’autre.
        outputs = model(batch_X)#On fait passer les entrées dans le réseau FFNN pour obtenir les prédictions outputs.
        loss = criterion(outputs, batch_Y)#On calcule la perte MSE entre la prédiction outputs et la vraie valeur batch_Y.
        loss.backward()#Backpropagation : calcul des gradients des poids du modèle par rapport à la perte.Chaque poids reçoit l’information sur comment le modifier pour réduire la perte.
        optimizer.step()#L’optimiseur (ici Adam) met à jour les poids du modèle en utilisant les gradients calculés.
        total_loss += loss.item()#On additionne la perte du batch au total de l’époque (loss.item() convertit le tenseur PyTorch en float Python).
    loss_list.append(total_loss)#On enregistre la perte totale de l’époque dans loss_list.
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")#On affiche la perte pour suivre l’évolution de l’entraînement.

end_time = time.time()  # fin du training
print(f"\n=== Training finished in {end_time - start_time:.2f} seconds ===")
# === Tracé du loss ===
plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Courbe de perte (training loss)")
plt.grid()
plt.show()

# === Sauvegarde du modèle et des paramètres de normalisation ===
save_path_model = os.path.join(script_dir, 'ffnn_model.pth')
torch.save(model.state_dict(), save_path_model)

# Sauvegarder les paramètres de normalisation
np.savez(os.path.join(script_dir, 'normalization_params.npz'),
         Minput_min=Minput_min,
         Minput_max=Minput_max,
         Moutput_min=Moutput_min,
         Moutput_max=Moutput_max)