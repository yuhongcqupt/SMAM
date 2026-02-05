import scipy.io
import numpy as np

import torch



def create_adjacency_matrix_cooccurance(data_label):
    cooccur_matrix = np.zeros((data_label.shape[1], data_label.shape[1]), dtype=float)
    for y in data_label:
        y = list(y)
        for i in range(len(y)):
            for j in range(len(y)):
                # data_label
                if y[i] == 1 and y[j] == 1:
                    cooccur_matrix[i, j] += 1
    row_sums = data_label.sum(axis=0)

    for i in range(cooccur_matrix.shape[0]):
        for j in range(cooccur_matrix.shape[0]):
            if row_sums[i] != 0:
                cooccur_matrix[i][j] = cooccur_matrix[i, j] / row_sums[i]
            else:
                cooccur_matrix[i][j] = cooccur_matrix[i, j]

    return cooccur_matrix


def gen_adj(A):

    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj




