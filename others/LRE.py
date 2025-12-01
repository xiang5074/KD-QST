# -*- coding: utf-8 -*-
# @Date:   2025-12-01 
# @Author: foxwy
# @Method: linear regression estimation, recursive LRE algorithm
# @Paper: Unifying the factored and projected gradient descent for quantum state tomography

# Linear regression estimation utilities for quantum state tomography with Pauli measurements


import sys
import numpy as np
import torch
from time import perf_counter

#torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=8)

sys.path.append('../..')
from Basis_Function import (qmt_torch, 
                                  qmt_matrix_torch, 
                                  proj_spectrahedron, 
                                  qmt_product_torch, 
                                  ten_to_k, 
                                  cal_Fidelity)


def cal_para(X, Y, n_qubits):
    """Using the product structure of POVM to speed up"""
    N = 5
    K = 6
    if n_qubits <= N:  # faster, further expansion will cause the memory to explode
        X_t = X
        for i in range(n_qubits - 1):
            X_t = torch.kron(X_t, X)
        return X_t @ Y
    else:
        Y = Y.reshape(-1, K**N)
        n_qubits_t = n_qubits - N
        N_choice = K**(n_qubits_t)
        num_choice = np.arange(N_choice)
        theta = 0

        X_t = X
        for i in range(N - 1):
            X_t = torch.kron(X_t, X)

        for num in num_choice:
            samples = ten_to_k(num, K, n_qubits_t)  
                # Example: if samples = [2, 0, 3], choose column 2 of the first qubit, column 0 of the second, column 3 of the third
            theta_n = X[:, samples[0]]
            for sample in samples[1:]:
                theta_n = torch.kron(theta_n, X[:, sample])
            theta_n = torch.kron(theta_n, X_t @ Y[num, :])   #  Y[num, :] selects the num-th row of Y
            theta += theta_n

        return theta


def LRE(M, n_qubits, P_data, M_basis, device='cpu', data_type=torch.complex128):

    if data_type == torch.complex64:
        data_type1 = torch.float32
        data_type2 = torch.complex64
    elif data_type == torch.complex128:
        data_type1 = torch.float64
        data_type2 = torch.complex128
        
    time_b = perf_counter()

    X = qmt_product_torch([M], [M_basis])
    X_t = torch.linalg.pinv(X.T @ X, hermitian=True)
    X_t = X_t @ X.T

    # method2
    P_data = P_data.to(data_type1)
    #P_data = P_data.to(torch.float64)
    #X_t = X_t.to(torch.float64)
    theta = cal_para(X_t, P_data, n_qubits)
    #rho = qmt_matrix_torch(theta.to(torch.complex64), [M_basis.to(torch.complex64)] * n_qubits)
    rho = qmt_matrix_torch(theta.to(data_type2), [M_basis] * n_qubits)

    # state-mapping
    rho = proj_spectrahedron(rho)

    rho = rho / torch.trace(rho)

    time_e = perf_counter()
    time_all = time_e - time_b

    return rho, time_all




if __name__ == '__main__':
    pass
