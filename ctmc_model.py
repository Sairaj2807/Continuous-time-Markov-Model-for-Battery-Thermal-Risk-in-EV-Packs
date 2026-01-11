import numpy as np
from scipy.linalg import expm

class CTMC:
    def __init__(self, Nij, Ti):
        self.Nij = Nij
        self.Ti = Ti
        self.Q = self.build_rate_matrix()

    def build_rate_matrix(self):
        Q = np.zeros((4, 4))

        for i in range(4):
            for j in range(4):
                if i != j:
                    Q[i, j] = self.Nij[i, j] / self.Ti[i] if self.Ti[i] > 0 else 0
        
        for i in range(4):
            Q[i, i] = -np.sum(Q[i, :])

        Q[3] = [0, 0, 0, 0]  # S3 absorbing state
        
        return Q

    def predict_probs(self, initial_state, t_seconds):
        P0 = np.zeros(4)
        P0[initial_state] = 1
        P_t = P0 @ expm(self.Q * t_seconds)
        return P_t

    def compute_TRI(self, prob):
        w = np.array([0, 0.3, 0.7, 1.0])
        return float(np.dot(prob, w))
