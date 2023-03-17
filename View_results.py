import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_config_dict(nom):
    """
    Loads config dict (pickle) saved from main.py
    """
    path = 'D:\DEVOIRS\AA\Projet_AA/'
    with open(path+nom, 'rb') as handle:
        return(pickle.load(handle))

di = load_config_dict('7_accuracy_10')
print(di)
plt.plot(di['graph'])
plt.show()

