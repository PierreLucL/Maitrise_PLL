import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from Sampler import sampler
from Classe_posters import Posters
from Fonctions_utiles import custom_loss2, compute_accuracy
from Classe_réseau import AlexNet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm
import time 
import pickle

def save_dict(config_dict,compteur, name):
    title = f'{compteur}_accuracy_{name}'
    title2 = f'D:/DEVOIRS\AA\Projet_AA/{title}'
    with open(title2, 'wb') as handle:
        pickle.dump(config_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Faire les calculs sur le GPU plutôt que le CPU si possible
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Vide la cache de la GPU
torch.cuda.empty_cache()

# Setup du .csv dela base de données pour avoir les genres
df = pd.read_csv("Multi_Label_dataset/train.csv")

# Path relatif des images
img_folder = 'Multi_Label_dataset/Images'

# Standardisation des images à effectuer
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor()])

# Instanciation de l'ensemble des données
whole_dataset = Posters(df, img_folder, transform)


sample, not_sample = sampler(whole_dataset)
new_df = df.drop(sample)
wasted_df = df.drop(not_sample)

train_set, test_set = train_test_split(new_df,test_size=300,train_size=3200)
# Initialisation de data d'entraînement et de validation
train_dataset = Posters(train_set, img_folder, transform)
test_dataset = Posters(test_set, img_folder, transform)

# Création du dataloader d'entraînement et de validation
train_dataloader = DataLoader(
    train_dataset,batch_size=100, shuffle=True)
test_dataloader = DataLoader(
    test_dataset,batch_size=100, shuffle=True)

criterions = [nn.BCELoss(), custom_loss2]
batch = 100
epoch = 120
learning_rate = [0.1, 0.06, 0.02]
momentum = [0.01,0.1]

model = AlexNet()
model.to(DEVICE)

compteur = 0
model.train()
for critere in criterions:
    for lr in learning_rate:
        for mom in momentum:
            compteur += 1
            exp = {}
            optimizer = SGD(model.parameters(), lr=lr,momentum=mom)

            accuracy_plot = []
            all_losses = []
            start = time.time()
            # Boucle d'entraînement / Training loop
            for i_epoch in range(epoch):
                # On boucle sur ce qui se trouve dans le dataloader 
                loss_epoch = []
                for i_batch, batch in enumerate(tqdm(train_dataloader)):
                    
                    # Setup des images et de leurs targets
                    images , targets = batch['image'], batch['labels']
                    targets = targets.type(torch.FloatTensor)
                    images = images.to(DEVICE)
                    targets = targets.to(DEVICE)

                    # Mettre les gradients à zéro
                    optimizer.zero_grad()

                    # Calcul de l'inférence dans une variable "predictions"
                    # et calcul de l'erreur dans une variable "loss"
                    predictions = model(images)
                    loss = critere(predictions, targets)
                    all_losses.append(loss)
                    #tqdm.write(str(loss))

                    # Rétropropagation de l'erreur et optimisation
                    loss.backward()
                    optimizer.step()
                accuracy, w,t,v, test_loss = compute_accuracy(model, test_dataloader)
                accuracy_plot.append(compute_accuracy(model, test_dataloader)[0])
            end = time.time()
            exp["temps de calcul"] = end-start
            exp['accuracy'] = accuracy_plot[-1]
            exp['momentum'] = mom
            exp['accuracy_graph'] = accuracy_plot
            exp['train_loss_graph'] = all_losses
            exp['test_loss_grap'] = all_test_losses
            exp['learning rate'] = lr
            exp['criterion'] = critere
            name =  int(accuracy_plot[-1]*100)
            save_dict(exp,compteur,name)
            torch.cuda.empty_cache()

