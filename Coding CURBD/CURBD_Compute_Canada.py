##################################### PREUVE DE CONCEPT CURB À COMPUTE CANADA #####################################
# Par Pierre-Luc Larouche
###################################################################################################################

# Import des librairie importantes
import numpy as np
from tqdm import tqdm
from numba import njit, prange


# La classe qui génère les données
class GeneratorModel3R:
    def __init__(self, N: int, g_AB: tuple, w_rgn: float, p_rgn: float):
        self.N = N  # Nombre de neurones
        self.tau = 0.1  # Pas de temps
        self.w_in = 1.0  #
        self.p_rgn = p_rgn

        self.J = np.empty((3 * self.N, 3 * self.N))
        for i in range(3):
            for j in range(3):
                if i == j:
                    dist = g_AB[i] * np.random.normal(loc=0.0, scale=np.sqrt(g_AB[i] ** 2 / (self.N)),
                                                      size=(self.N, self.N))
                else:
                    dist = w_rgn * np.random.choice([0.0, 1.0], size=(self.N, self.N), p=[1 - self.p_rgn, self.p_rgn])
                self.J[i * self.N:(i + 1) * self.N, j * self.N:(j + 1) * N] = dist
        C_SA = np.zeros((self.N, 1))
        C_SB = np.random.choice([0.0, -1.0], size=(self.N, 1), p=[1 - 0.5, 0.5])
        C_SC = np.random.choice([0.0, 1.0], size=(self.N, 1), p=[1 - 0.5, 0.5])
        self.C_S = self.w_in * np.concatenate((C_SA, C_SB, C_SC))

        self.r = np.random.uniform(low=-1.0, high=1.0, size=(3 * self.N, 1))

    def drdt(self, S: np.array):
        dr = 1 / self.tau * (-self.r + self.J @ np.tanh(self.r) + self.C_S * S)
        return dr

    def signal(self, t: float):
        def SQ(t: float):
            indices = np.arange(self.N).reshape(self.N, 1)
            num = (indices - int(self.N / 5) - self.N * t / 10) ** 2
            den = 2 * (int(self.N / 5)) ** 2
            return np.exp(-num / den)

        if t < 2.0:
            S_B = SQ(2.0)
            S_C = SQ(2.0)
        elif t < 6.0:
            S_B = SQ(t)
            S_C = SQ(2.0)
        elif t < 8.0:
            S_B = SQ(6.0)
            S_C = SQ(2.0)
        else:
            S_B = SQ(6.0)
            S_C = SQ(5.0)

        S = np.concatenate((np.full((self.N, 1), 0.0), S_B, S_C))
        return S

    def integrate(self, time: np.array):
        h = time[1] - time[0]
        rA = np.empty((self.N, len(time)))
        rB = np.empty((self.N, len(time)))
        rC = np.empty((self.N, len(time)))
        for i, t in enumerate(time):
            S = self.signal(t)
            dr = self.drdt(S)
            self.r = self.r + h * dr
            rA[:, i] = self.r[:self.N, 0]
            rB[:, i] = self.r[self.N:2 * self.N, 0]
            rC[:, i] = self.r[2 * self.N:, 0]
        return rA, rB, rC


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

    return x_vector, x_list, J, J_mean, P


N_list = [600,800,1000]
time = np.arange(0.0, 12.0, 0.01)
for idx, N_used in enumerate(N_list) :
    # On lance ensuite l'intégration du modèle
    model = GeneratorModel3R(N=N_used, g_AB=[1.8, 1.5, 1.5], w_rgn=0.01, p_rgn=0.01)
    rA, rB, rC = model.integrate(time=time)
    print('did it')

    # On définie les paramètres du modèle
    N = 3 * N_used # Nombre de neurones
    tau = 0.1 # Pas de temps
    alpha = 1.0 # Valeur du bruit initial sur la diagonale de P
    g = 1.5 # Constante de force des connections récurentes
    J_list = []


    # Ensuite quelques vecteurs et matrices
    J = np.random.normal(loc=0.0, scale=g/np.sqrt(N), size=(N, N))
    P = 1/alpha * np.identity(n=N)
    x_vector = np.random.uniform(low=-1.0, high=1.0, size=(N, 1))
    h = np.random.uniform(low=-1.0, high=1.0, size=(N, 1))

    # On défini le teacher
    teacher = np.concatenate((np.tanh(rA), np.tanh(rB), np.tanh(rC)), axis=0)

    for i in tqdm(range(12)):
        x_vector, x_list, J, J_mean, P = learn(N, x_vector, time, teacher, tau, g, J, h, P)
        J_list.append(J_mean)

    np.save(fr'/home/pllar11/scratch/model.J_{N_used}-{idx}.npy', model.J)
    np.save(fr'/home/pllar11/scratch/teacher_{N_used}-{idx}.npy', teacher)
    np.save(fr'/home/pllar11/scratch/J_list_{N_used}-{idx}.npy', J_list)
    np.save(fr'/home/pllar11/scratch/J_{N_used}-{idx}.npy', J)
    np.save(fr'/home/pllar11/scratch/x_list_{N_used}-{idx}.npy', x_list)


    # Ensuite quelques vecteurs et matrices
    J = np.random.normal(loc=0.0, scale=g/np.sqrt(N), size=(N, N))
    P = 1/alpha * np.identity(n=N)
    x_vector = np.random.uniform(low=-1.0, high=1.0, size=(N, 1))
    h = np.random.uniform(low=-1.0, high=1.0, size=(N, 1))
    J_list = []
    
    for i in tqdm(range(12)):
        x_vector, x_list, J, J_mean, P = learn(N, x_vector, time, teacher, tau, g, J, h, P)
        J_list.append(J_mean)

    np.save(fr'/home/pllar11/scratch/model.J_{N_used}-{idx+10}.npy', model.J)
    np.save(fr'/home/pllar11/scratch/teacher_{N_used}-{idx+10}.npy', teacher)
    np.save(fr'/home/pllar11/scratch/J_list_{N_used}-{idx+10}.npy', J_list)
    np.save(fr'/home/pllar11/scratch/J_{N_used}-{idx+10}.npy', J)
    np.save(fr'/home/pllar11/scratch/x_list_{N_used}-{idx+10}.npy', x_list)