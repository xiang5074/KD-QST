# -*- coding: utf-8 -*-
# @Author: Xiang Li, foxwy
# @Date:   2025-12-01 
# @Paper 1:  Efficient quantum state tomography with two complementary measurements
# @Paper 2:  Direct Quantum State Tomography Based on Kirkwood-Dirac Quasiprobability

# Core numerical utilities for POVM-based tomography, fidelity, and shuffling helpers


import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import gc
import time
import tqdm
import numpy as np
#from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


N_threshold = 12
N_moniter = 14
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def monitor_memory():   # Clear caches and monitor memory usage
    # Force garbage collection to clear unused tensors
    gc.collect()
    torch.cuda.empty_cache()

    '''# Use pynvml to monitor total GPU memory
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # Default to GPU 0
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    print(f"[GPU] Total Memory: {mem_info.total / 1024 ** 3} GB")
    print(f"[GPU] Used Memory: {mem_info.used / 1024 ** 3} GB")
    print(f"[GPU] Free Memory: {mem_info.free / 1024 ** 3} GB")'''


def cal_Fidelity(rho1, rho2):  
    #rho1 = rho1.to(dtype=torch.complex128)
    #rho2 = rho2.to(dtype=torch.complex128)
    N = rho1.shape[0]

    t0 = time.time()
    rho_tmp = rho1 @ rho2
    del rho1, rho2
    t1 = time.time()
    if N >= 2**N_moniter:
        print('-----fq rho_tmp-----, time=', t1-t0)
        monitor_memory()

    eigenvalues = torch.linalg.eigvals(rho_tmp)   # torch >= 1.10
    del rho_tmp
    if N >= 2**N_moniter:
        t2 = time.time()
        print('-----fq eigvals-----, time=', t2-t1)
        monitor_memory()

    eigenvalues = eigenvalues.real
    eigenvalues.clamp_(min=0)
    sqrt_eigvals = torch.sqrt(eigenvalues)
    Fq = torch.sum(sqrt_eigvals) ** 2

    Fq = torch.clamp(Fq, 0, 1-1e-8)   
    return Fq #.real


def proj_spectrahedron(rho_g):   
    #print('rho_g:', rho_g)
    rho_g = rho_g.to(device=device)
    rho_g = (rho_g + rho_g.conj().T) / 2   # Hermitian projection
    N = rho_g.shape[0]

    if N >= 2**N_moniter:
        print('\n-----01------')
        monitor_memory()
    
    # -------torch eigen decomposition---------
    vals, vecs = torch.linalg.eigh(rho_g)    
    u, _ = torch.sort(vals, descending=True)
    csu = torch.cumsum(u, dim=0)
    t = (csu - 1) / torch.arange(1, len(u)+1, device=device)
    nonzero_indices = torch.nonzero(u > t, as_tuple=False)
    del u,csu    # Delete variables to free memory

    idx_max = nonzero_indices[-1].item()
    tau = t[idx_max]  
    del t    # Delete variables to free memory

    vals.sub_(tau).clamp_(min=0)    
    rho_proj = vecs @ torch.diag(abs(vals)).to(dtype=vecs.dtype) @ vecs.conj().T
    rho_proj = 0.5 * (rho_proj + rho_proj.conj().T) 
    rho_proj = rho_proj / torch.trace(rho_proj)

    return rho_proj



def ten_to_k(num, k, N) -> list:
    """
    Convert decimal ``num`` to ``k`` decimal and complementary

    Args:
        num: Decimal numbers.
        k: k decimal.
        N: Total number of digits.

    Returns:
        Converted k decimal list.

    Examples::
        >>> ten_to_k(10, 2, 5)
        >>> [0, 1, 0, 1, 0]
        >>> ten_to_k(10, 4, 5)
        >>> [0, 0, 0, 2, 2]
    """
    transfer_num = []
    if num > k**N - 1:  # error num
        print('please input the right number!')
    else:
        while num != 0:
            num, a = divmod(num, k)
            transfer_num.append(a)
        transfer_num = transfer_num[::-1]
        if len(transfer_num) != N:
            transfer_num = [0] * (N - len(transfer_num)) + transfer_num
    return transfer_num


def shuffle_sample(p_all):
    # Infer N from the number of columns
    cols = p_all.shape[1]
    N = int(torch.log2(torch.tensor(cols)).item())
    
    # Initialize permutation indices
    permutation = []
    
    # Recursively generate block indices
    def generate_indices(rows_start, rows_end, cols_start, cols_end, level):
        if level == 0:
            # Base case: record flat indices
            for i in range(rows_start, rows_end):
                for j in range(cols_start, cols_end):
                    flat_idx = i * p_all.shape[1] + j
                    permutation.append(flat_idx)
        else:
            # Split rows into 3 groups and columns into 2 blocks
            row_step = (rows_end - rows_start) // 3
            col_step = (cols_end - cols_start) // 2
            
            # Process sub-blocks in order: group0/block0 → group0/block1 → group1/block0 → group1/block1 → group2/block0 → group2/block1
            for g in [0, 1, 2]:
                for c in [0, 1]:
                    new_rows_start = rows_start + g * row_step
                    new_rows_end = new_rows_start + row_step
                    new_cols_start = cols_start + c * col_step
                    new_cols_end = new_cols_start + col_step
                    generate_indices(new_rows_start, new_rows_end, new_cols_start, new_cols_end, level-1)
    
    # Kick off recursion over the full matrix with depth N
    generate_indices(0, p_all.shape[0], 0, p_all.shape[1], N)
    
    # Convert to tensor and apply permutation
    permutation = torch.tensor(permutation)
    q_reordered = p_all.flatten()[permutation].real.view(-1, 1)
    
    return q_reordered




def shuffle_forward_torch(rho, dims):
    """
    To transpose the density matrix, see
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [torch] version we implemented.
    """
    N = len(dims)
    rho = rho.T
    rho = rho.reshape(tuple(torch.cat([dims, dims], 0)))
    #ordering = torch.reshape(torch.arange(2*N).reshape(2, -1).T, (1, -1))[0]
    ordering = torch.arange(2 * N).reshape(2, -1).T.flatten()
    rho = rho.permute(tuple(ordering))
    return rho


def qmt_torch(X, operators, allow_negative=True):
    """
    Simplifying the computational complexity of mixed state measurements using the product structure of POVM, see 
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [torch] version we implemented.

    Args:
        X (tensor): Density matrix.
        operators (list, [tensor]): The set of single-qubit measurement, such as [M1, M2, ....], the size of M is (k, 2, 2).
        allow_negative (bool): Flag for whether to set the calculated value negative to zero.
    """
    if not isinstance(operators, list):
        operators = [operators]

    N = len(operators)  # qubits number
    Ks = torch.zeros(N, dtype=torch.int)
    Ds = torch.zeros(N, dtype=torch.int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))  # reshape: row-major verctorization
    del Ks

    if N > N_threshold:  # torch does not support more dimensional operations
        X = X.cpu()
    X = shuffle_forward_torch(X, Ds)
    X = X.reshape(-1, Ds[-1]**2)     
    if N > N_threshold:
        X = X.to(operators[0].device)

    for i in range(N - 1, -1, -1):
        P = operators[i]
        X = torch.matmul(P, X.T)
        if i > 0:
            X = X.reshape(-1, Ds[i]**2)


    #if self.mea_flag != 'kd':
    #    P_all = torch.real(X.reshape(-1))  
    P_all = X.reshape(-1,1)

    if not allow_negative:
        P_all = torch.maximum(P_all, torch.tensor(0))
        P_all /= torch.sum(P_all)
    return P_all




def shuffle_adjoint_torch(R, dims):
    """
    To transpose the density matrix, see
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [torch] version we implemented.
    """
    N = len(dims)
    R = R.reshape(tuple(torch.cat([dims, dims], 0)))
    ordering = torch.arange(2 * N).reshape(-1, 2).T.reshape(-1)
    R = R.permute(tuple(ordering))
    R = R.reshape(torch.prod(dims), torch.prod(dims))

    return R


def reorder_g(g, dims):
    """
    Reorder g based on the shuffle pattern used in shuffle_adjoint_torch to match the correct matrix product order.
    """
    N = len(dims)
    g = g.reshape(tuple(torch.cat([dims, dims], 0)))
    ordering = torch.arange(2 * N).reshape(-1, 2).T.reshape(-1)
    g = g.permute(tuple(ordering))
    return g.flatten()

def qmt_matrix_torch(X, operators):
    """
    Simplifying the computational complexity of mixed state measurement operator mixing using the product structure of POVM, see 
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [torch] version we implemented.

    Examples::
        >>> M = torch.tensor([a, b, c, d])
        >>> qmt_matrix(torch.tensor([1, 2, 3, 4]), [M])
        >>> 1 * a + 2 * b + 3 * c + 4 * d
        a, b, c, d is a matrix.
    """
    if not isinstance(operators, list):
        operators = [operators]

    N = len(operators)  # qubits number
    Ks = torch.zeros(N, dtype=torch.int)
    Ds = torch.zeros(N, dtype=torch.int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))

    #if N > 12:  # torch does not support more dimensional operations
    #    X = X.cpu()
    #X = shuffle_forward_torch(X, Ds)   # Adjust T so g aligns with the [M]*N product order used in POVM calculations

    for i in range(N - 1, -1, -1):
        P = operators[i]
        X = X.reshape(-1, Ks[i])

        X = torch.matmul(X, P)
        X = X.T

    if N > N_threshold:  # torch does not support more dimensional operations
        X = X.to('cpu')
    X = shuffle_adjoint_torch(X, Ds.flip(dims=[0]))
    if N > N_threshold:
        X = X.to(operators[0].device)
    #X = 0.5 * (X + X.T.conj())
    return X


def qmt_product_torch(operators_1, operators_2):
    """
    To calculate the X matrix in the LRE algorithm, see paper ``Full reconstruction of a 
    14-qubit state within four hours```.
    """
    if not isinstance(operators_1, list):
        operators_1 = [operators_1]

    if not isinstance(operators_2, list):
        operators_2 = [operators_2]

    N = len(operators_1)  # qubits number
    Ks_1 = torch.zeros(N, dtype=torch.int)
    Ds_1 = torch.zeros(N, dtype=torch.int)
    Ks_2 = torch.zeros(N, dtype=torch.int)
    Ds_2 = torch.zeros(N, dtype=torch.int)
    for i in range(N):
        dims = operators_1[i].shape
        Ks_1[i] = dims[0]
        Ds_1[i] = dims[1]
        dims = operators_2[i].shape
        Ks_2[i] = dims[0]
        Ds_2[i] = dims[1]

    operators_t = torch.einsum('...ij->...ji', [operators_1[0]])  # Transpose the last two dimensions
    P_single = torch.real(torch.matmul(operators_t.reshape(Ks_1[0], Ds_1[0]**2), operators_2[0].reshape(Ks_2[0], Ds_2[0]**2).T))  
                    # If operators1 elements are scalars, this is elementwise scaling
    X_t = P_single
    for i in range(N - 1):
        X_t = torch.kron(X_t, P_single)
    return X_t



def Is_rho(rho):
    """
    Determine if ``rho`` is a density matrix.

    Density matrix properties:
        1. unit trace.
        2. semi-definite positive.
        3. Hermitian.
    """
    if abs(torch.trace(rho).item() - 1) > 1e-7:
        print('trace of rho is not 1')
        return 0
    elif torch.all(torch.abs(rho - rho.T.conj()) > 1e-8):
        print('rho is not Hermitian') 
        return 0
    elif not semidefinite_adjust(rho):
        print('rho is not positive semidefine')
        return 0
    else:
        return 1

    
def semidefinite_adjust(M, eps=1e-9):
    """
    Determine whether the matrix ``M`` is a semi-positive definite matrix.

    Returns:
        bool: True if ``M`` is a semi-positive definite matrix, otherwise False.
    """
    M_vals, M_vecs = torch.linalg.eigh(M)
    #print('eigenvalues2:', M_vals)
    if torch.all(M_vals > -eps):
        return True
    else:
        print('smallest eigenvalues:', M_vals[:5])
        return False
    
def safe_dot_product(vec1, vec2, chunk_size=2**30):
    """Chunked dot product that safely handles extremely large vectors."""    
    total = torch.tensor(0.0, device=vec1.device, dtype=vec1.dtype)
    for i in range(0, vec1.numel(), chunk_size):
        chunk1 = vec1[i:i+chunk_size]
        chunk2 = vec2[i:i+chunk_size]
        total += torch.dot(chunk1, chunk2)
    return total


#--------------------main--------------------
if __name__ == '__main__':
    pass   
