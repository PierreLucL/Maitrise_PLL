import torch
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import pandas as pd
from torchvision import transforms
import time


df = pd.read_csv("Multi_Label_dataset/train.csv")



def create_balanced_sampler(dataset):
    def make_weights_for_balanced_classes(images, n_classes=25):
        count = np.array([0] * n_classes)
        for item in images:
            count[np.transpose(np.argwhere(item == 1))[0]] += 1
        weight_per_class = [0.] * n_classes
        N = float(sum(count))
        for i in range(n_classes):
            weight_per_class[i] = N/float(count[i])
        weight = [0] * len(images)
        for j, val in enumerate(images):
            weight[j] = weight_per_class[val[1]]
        return weight
    weights = make_weights_for_balanced_classes(dataset.labels)
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler

def elagueur(dataset):
    total = np.zeros(25)
    for labels in dataset.labels:
            total += labels
    somme = np.sum(total)
    ratio = total/somme
    useless_col = []
    for i,rat in enumerate(ratio):
        if rat < 0.02:
            useless_col.append(i)
    elagage = np.zeros(25)
    elagage[useless_col] = 1
    indices = []
    for indice, labels in enumerate(dataset.labels):
        cheker = labels*elagage
        if np.sum(cheker)> 0:
            indices.append(indice)
    #print(indices,len(indices))
    return indices


def sampler(dataset, elag):
    total = np.zeros(25)
    total2 = np.zeros(25)
    indices = elag
    not_indices = []
    for indice, labels in enumerate(dataset.labels):
        total += labels
        cheker = labels * total
        cheker = cheker>1000
        if sum(cheker) != 0:
            indices.append(indice)
            total -= labels
        else:
            not_indices.append(indice)
    indices = list(dict.fromkeys(indices))
    return indices, not_indices
    
