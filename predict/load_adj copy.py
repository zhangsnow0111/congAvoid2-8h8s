import numpy as np
import torch
from math import sqrt
import pandas

def load_adj(switchNumber = 8, feature_len = 8):
    print('Loading data...')
    adj = pandas.read_csv("./predict/adj.csv", header=None)
    adj = np.array(adj)
    # A~ = A + I

    # D_A_final = D_hat**-1/2 * A_hat *D_hat**-1/2
    I = np.matrix(np.eye(8))
    A_hat = adj + I
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat_sqrt = [sqrt(x) for x in D_hat]
    D_hat_sqrt = np.array(np.diag(D_hat_sqrt))
    D_hat_sqrtm_inv = np.linalg.inv(D_hat_sqrt)  # get the D_hat**-1/2 (开方后求逆即为矩阵的-1/2次方)
    D_A_final = np.dot(D_hat_sqrtm_inv, A_hat)
    D_A_final = np.dot(D_A_final, D_hat_sqrtm_inv)
    D_A_final = torch.tensor(D_A_final).float()
    
    return D_A_final