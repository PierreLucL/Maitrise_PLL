# Projet de maîtrise #
# Par Pierre-Luc Larouche #

#==================//=======================#
# Importation des différents modules utiles
#==================//=======================#
import numpy as np
import torch.nn as nn
import torch
from torch.optim import SGD
from torchvision import datasets, transforms, utils
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset, dataset
from tqdm import tqdm
from torchvision.models import resnet18

#==================//=======================#
# Importation des différentes classes implémentées
#==================//=======================#
from Classe_réseau import PostersNet, PostersNet2, AlexNet, ResNet18
from Fonctions_utiles import compute_accuracy, plot_hist, plot_results, custom_loss, custom_loss2, accuracy_random
from Sampler import sampler, elagueur


# Faire les calculs sur le GPU plutôt que le CPU si possible
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Vide la cache de la GPU
torch.cuda.empty_cache()

# Path relatif des images
img_folder = 'Multi_Label_dataset/Images'

# Standardisation des images à effectuer
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor()])

# Instanciation de l'ensemble des données
whole_dataset = Posters(df, img_folder, transform)

# Élagage des genres de fims sous-représentés dans le dataset
elagage = elagueur(whole_dataset)

# Sélection de la meilleure distribution possible des genres pour l'entraîenement
sample, not_sample = sampler(whole_dataset, elagage)
new_df = df.drop(sample)

# Initialisation des noms des images élagées et sélectionnées
train_set, test_set = train_test_split(new_df,test_size=301,train_size=2500)

# Initialisation de data d'entraînement et de validation
train_dataset = Posters(train_set, img_folder, transform)
test_dataset = Posters(test_set, img_folder, transform)

# Initialisation des paramètres d'entraînement
largeur = 300
nb_epoch = 3
batch_size = 10
learning_rate = 0.00335
momentum = 0
hist = True

# Création du dataloader d'entraînement et de validation
train_dataloader = DataLoader(
    train_dataset,batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(
    test_dataset,batch_size=batch_size, shuffle=True)

# Graphique de l'histogramme des données d'entraînement et de validation
if hist:
    plot_hist(train_dataloader)
    plot_hist(test_dataloader)
    plt.show()

# Initialisation du réseau
model = ResNet18()
model.to(DEVICE)

# On instancie une fonction d'erreur BinaryCrossEntropy qui sera le critère à optimimiser
#criterion = nn.BCELoss()
criterion = custom_loss2
#criterion = nn.CrossEntropyLoss()


# On instance l'algorithme d'optimisation SGD avec les hyperparamètres
params = filter(lambda x: x.requires_grad, model.parameters())
#optimizer = SGD(model.parameters(), lr=learning_rate,
#momentum=momentum)
optimizer = SGD(params, lr=learning_rate,
momentum=momentum)

# On met le modèle en mode entraînement
model.train()
accuracy_plot = []
train_loss_plot =[]
test_loss_plot = []

# Boucle d'entraînement / Training loop
for i_epoch in range(nb_epoch):
    # Initiation de la liste d'erreur
    train_losses = []

    # On boucle sur ce qui se trouve dans le dataloader 
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
        loss = criterion(predictions, targets)
        train_losses.append(loss.item())

        # Rétropropagation de l'erreur et optimisation
        loss.backward()
        optimizer.step()

        # Accumulation du loss de la batch
        train_losses.append(loss.item())
    train_loss_plot.append(np.mean(np.array(train_losses)))
    # Imprimer après chaque époque
    print(' [-] epoch {:4}/{:}, train loss {:.6f}'.format(
        i_epoch + 1, nb_epoch, np.mean(train_losses)))
    
    accuracy, w,t,v, test_loss = compute_accuracy(model, test_dataloader, criterion) 
    accuracy_plot.append(accuracy)
    test_loss_plot.append(test_loss)
    print('accuracy : ', accuracy, '\n', 'test_loss : ', test_loss)

# Affichage du score en test / Display test score
test_acc ,  predictions, targets,all_images, test_loss = compute_accuracy(model, test_dataloader, criterion)
print(' [-] test acc. {:.6f}%'.format(test_acc * 100))
plot_results(predictions, targets,all_images,  15)

#Libère le gpu
torch.cuda.empty_cache()

plt.plot(accuracy_plot, color='red', label='Accuray test')
plt.plot(test_loss_plot, linestyle='--', color='red', label='Test loss')
plt.plot(train_loss_plot, linestyle='--', color='blue', label='Train loss')
plt.legend()
plt.title(f'{nb_epoch} epoch, learning rate ={learning_rate}, momentum={momentum}')
plt.show()
