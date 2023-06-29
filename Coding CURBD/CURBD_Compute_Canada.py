##################################### PREUVE DE CONCEPT CURB À COMPUTE CANADA #####################################
# Par Pierre-Luc Larouche
###################################################################################################################

# Import des librairie importantes
import numpy as np
from tqdm import tqdm
from numba import njit, prange

# Toutes les fonctions décorées nécessaires au learning
@njit
def dxdt(tau, x_vector, g, J, h):
    return 1/tau * (-x_vector + g * J @ np.tanh(x_vector) + h)

@njit
def dhdt(N,h):
    eta = np.random.normal(loc=0.0, scale=1.0, size=(N, 1))
    return 1/0.1 * (-h + eta)

@njit
def update_J(x, P, J, f):
    cte = (1 / (1 + np.tanh(x).T @ P @ np.tanh(x)))[0][0]
    error = np.tanh(x) - f

    P = P - P @ np.tanh(x) @ np.tanh(x).T @ P * cte
    J = J - cte * error @ (P @ np.tanh(x)).T

    return P, J

@njit
def learn(N, x_vector, time, signal, tau, g, J, h, P):
    # Initialisation de différentes matrices vides et constantes
    dt = time[1] - time[0]
    J_mean = np.empty(len(time))
    x_list = np.empty((N, len(time)))
    f = np.empty((N, 1))

    for i in range(len(time)):

        f[:, 0] = signal[:, i]
        h = h + dt * dhdt(N, h)
        x_vector = x_vector + dt * dxdt(tau, x_vector, g, J, h)

        x_list[:, i] = x_vector[:, 0]

        if not (i + 1) % 2:
            P, J = update_J(x_vector, P, J, f)

        a = np.abs(J)
        J_mean[i] = a.mean()

    return x_list, J, J_mean, P

# On lance ensuite l'intégration du modèle
time = np.arange(0.0, 12.0, 0.01)
#teacher = np.load(r'/home/pllar11/scratch/teacher_officiel_300.npy')
teacher = np.load(r"teacher_simple.npy")
etat_initial = teacher[:, 0]
x_vector = np.reshape(etat_initial, (300, 1))

N_iterations = [1,2,3,4,5,10,40,70,100,130,160,190]

# On définie les paramètres du modèle
N = 300 # Nombre de neurones
tau = 0.1 # Pas de temps
alpha = 1.0 # Valeur du bruit initial sur la diagonale de P
g = 1.5 # Constante de force des connections récurentes
J_list = []


# Ensuite quelques vecteurs et matrices
J = np.random.normal(loc=0.0, scale=g/np.sqrt(N), size=(N, N))
P = 1/alpha * np.identity(n=N)
#x_vector = np.random.uniform(low=-1.0, high=1.0, size=(N, 1))
h = np.random.uniform(low=-1.0, high=1.0, size=(N, 1))

for i in tqdm(range(190)):
    x_list, J, J_mean, P = learn(N, x_vector, time, teacher, tau, g, J, h, P)
    J_list.append(J_mean)

    if i in [0,1,2,3,4,9,39,69,99,129,159,189]:
        #np.save(fr'/home/pllar11/scratch/J_list_300_1_it{i+1}.npy', J_list)
        #np.save(fr'/home/pllar11/scratch/J_300_1_it{i+1}.npy', J)
        #np.save(fr'/home/pllar11/scratch/x_list_300_1_it{i+1}.npy', x_list)
        np.save(fr'J_list_900_it{i + 1}.npy', J_list)
        np.save(fr'J_900_it{i+1}.npy', J)
        np.save(fr'x_list_900_it{i+1}.npy', x_list)