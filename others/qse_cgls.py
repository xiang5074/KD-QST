# -*- coding: utf-8 -*-
# @Date:   2025-12-01 
# @Author: foxwy
# @Method: conjugate-gradient algorithm
# @Paper: Unifying the factored and projected gradient descent for quantum state tomography

# Conjugate-gradient line-search solver for quantum state estimation


import sys
import numpy as np
import torch
from time import perf_counter
from tqdm import tqdm

sys.path.append('../..')
from Basis_Function import qmt_torch, qmt_matrix_torch, cal_Fidelity, safe_dot_product, monitor_memory



def qse_cgls(M, n_qubits, P_data, epochs, his_flag, result_save, data_type1, data_type2, device='cpu', rho_star=0):
    """
    conjugate-gradient algorithm, see paper
    ``Superfast maximum-likelihood reconstruction for quantum tomography``.
    
    Args:
        M (tensor): The POVM, size (K, 2, 2).
        n_qubits (int): The number of qubits.
        P_data (tensor): The probability distribution obtained from the experimental measurements.
        epochs (int): Maximum number of iterations.
        fid (Fid): Class for calculating fidelity.
        result_save (set): A collection that holds process data.
        device (torch.device): GPU or CPU. 

    Stops:
        Reach the maximum number of iterations or quantum fidelity greater than or equal to 0.99 or other conditions.

    Examples::
        see ``FNN/FNN_learn`` or main.
    """
    # parameter
    opts = {'adjustment': 0.5, 'mincondchange': -torch.inf, 'step_adjust': 2, 'a2': 0.1, 'a3': 0.2}

    d = torch.tensor(2**n_qubits)
    M = M.to(data_type2)
    P_data = P_data.to(data_type1).flatten()
    P_data = torch.clamp_min(P_data, 1e-16)
    eps = np.finfo(np.complex128).eps
    threshold_step = torch.sqrt(d) * d * eps
    threshold_fval = - safe_dot_product(P_data, torch.log(P_data))

    # rho init
    #A = (torch.eye(d) / torch.sqrt(d)).to(data_type2).to(device)
    #rho = (torch.eye(d) / d).to(data_type2).to(device)
    A = torch.randn(d, d).to(data_type2).to(device)
    rho = torch.matmul(A.T.conj(), A)
    rho = rho / torch.trace(rho)

    # -----line search stuff-----
    a2 = opts['a2']
    a3 = opts['a3']

    # discard zero-valued frequencies
    probs = qmt_torch(rho, [M] * n_qubits).to(data_type1).flatten()
    probs = torch.clamp_min(probs, 1e-16)
    adj = P_data / probs
    adj[P_data == 0] = 0
    if n_qubits >= 10:
        del probs  # free temporary tensor
        monitor_memory()

    rmatrix = qmt_matrix_torch(adj.to(data_type2), [M] * n_qubits)
    if n_qubits >= 10:
        del adj
        monitor_memory()

    condchange = torch.inf

    if opts['mincondchange'] > 0:
        hessian_proxy = P_data / probs**2  # Note: probs was deleted, but here it is recalculated below if needed.
        hessian_proxy[P_data == 0] = 0
        old_hessian_proxy = hessian_proxy

    fval = - safe_dot_product(P_data, torch.log(qmt_torch(rho, [M] * n_qubits).to(data_type1).flatten()))
    if n_qubits >= 10:
        monitor_memory()

    # iterative
    pbar = tqdm(range(epochs))
    time_all = 0
    stop_i = -1
    for i in pbar:
        time_b = perf_counter()

        curvature_too_large = False

        if opts['mincondchange'] > 0:
            if i > 0:
                condchange = torch.real(torch.acos(torch.real(safe_dot_product(old_hessian_proxy, hessian_proxy)) / torch.norm(old_hessian_proxy) / torch.norm(hessian_proxy)))

        # the gradient
        if i == 0:
            # gradient
            G = torch.matmul(A, rmatrix - torch.eye(d).to(device))
            # conjugate-gradient
            H = G
        else:
            G_next = torch.matmul(A, rmatrix - torch.eye(d).to(device))
            polakribiere = torch.real(torch.matmul(G_next.reshape(-1, 1).T.conj(), (G_next.reshape(-1, 1) - opts['adjustment'] * G.reshape(-1, 1)))) / torch.norm(G)**2
            gamma = torch.maximum(polakribiere, torch.tensor(0))
            # conjugate-gradient update and gradient assignment
            H = G_next + gamma * H
            G = G_next
            if n_qubits >= 10:
                del G_next  # free temporary gradient tensor
                monitor_memory()

        # line search
        A2 = A + a2 * H
        A3 = A + a3 * H
        rho2 = torch.matmul(A2.T.conj(), A2)
        rho2 = rho2 / torch.trace(rho2)
        rho3 = torch.matmul(A3.T.conj(), A3)
        rho3 = rho3 / torch.trace(rho3)
        probs2 = qmt_torch(rho2, [M] * n_qubits).to(data_type1).flatten()
        probs3 = qmt_torch(rho3, [M] * n_qubits).to(data_type1).flatten()
        probs2 = torch.clamp_min(probs2, 1e-16)
        probs3 = torch.clamp_min(probs3, 1e-16)

        l1 = fval
        l2 = - safe_dot_product(P_data, torch.log(probs2))
        l3 = - safe_dot_product(P_data, torch.log(probs3))
        if n_qubits >= 10:
            del probs2, probs3
            monitor_memory()
        alphaprod = 0.5 * ((l3 - l1) * a2**2 - (l2 - l1) * a3**2) / ((l3 - l1) * a2 - (l2 - l1) * a3)
        
        if torch.isnan(alphaprod) or alphaprod > 1 / eps or alphaprod < 0:
            candidates = [0, a2, a3]
            l_list = [l1, l2, l3]
            index = l_list.index(min(l_list))
            if opts['step_adjust'] > 1:
                if torch.isnan(alphaprod) or alphaprod > 1 / eps:
                    # curvature too small to estimate properly: enlarge step
                    a2 = opts['step_adjust'] * a2
                    a3 = opts['step_adjust'] * a3
                elif alphaprod < 0:
                    # curvature too large, steps overshoot parabola: reduce step
                    a2 = a2 / opts['step_adjust']
                    a3 = a3 / opts['step_adjust']
                    curvature_too_large = True

            alphaprod = candidates[index]
            if n_qubits >= 10:
                del candidates, l_list, index
                monitor_memory()

        # update A and normalize
        A = A + alphaprod * H
        A = A / torch.norm(A)
        # update old rho
        old_rho = rho

        # map
        T_temp = torch.matmul(A.T.conj(), A)
        rho = T_temp / torch.trace(T_temp)
        if n_qubits >= 10:
            del T_temp  # free temporary tensor from mapping step
            monitor_memory()

        probs = qmt_torch(rho, [M] * n_qubits).to(data_type1).flatten()
        probs = torch.clamp_min(probs, 1e-16)
        fval = - safe_dot_product(P_data, torch.log(probs))

        # check threshold
        steps_i = 0.5 * torch.sqrt(d) * torch.norm(rho - old_rho)
        satisfied_step = steps_i <= threshold_step
        satisfied_fval = fval <= threshold_fval

        if i < epochs - 1:
            adj = P_data / probs
            adj[P_data == 0] = 0
            rmatrix = qmt_matrix_torch(adj.to(data_type2), [M] * n_qubits)
            if n_qubits >= 10:
                del adj  # clear temporary tensor
                monitor_memory()
            if opts['mincondchange'] > 0:
                old_hessian_proxy = hessian_proxy
                hessian_proxy = P_data / probs**2
                hessian_proxy[P_data == 0] = 0

        time_e = perf_counter()
        time_all += time_e - time_b

        # evaluation
        if his_flag == 1:
            Fq = cal_Fidelity(rho, rho_star)
            result_save['Fq'].append(Fq.item())
            pbar.set_description("CG Fq {:.8f} | time {:.4f} | epochs {:d}".format(Fq, time_all, i + stop_i))     
            if len(result_save['Fq'])>=5:
                diffs = [abs(result_save['Fq'][i+1] - result_save['Fq'][i]) for i in range(len(result_save['Fq'])-5, len(result_save['Fq'])-1)]
                if i + 1 > epochs or max(diffs) <= 1e-3 or Fq.item() >= 0.999:
                    stop_i = i
                    break

        elif (i + 1) % 20 == 0 or i == 0:
            Fq = cal_Fidelity(rho, rho_star)
            result_save['Fq'].append(Fq.item())
            result_save['time'] = time_all
            pbar.set_description("CG Fq {:.8f} | time {:.4f} | epochs {:d}".format(Fq, time_all, i + 1))
            if len(result_save['Fq'])>=5:
                diffs = [abs(result_save['Fq'][i+1] - result_save['Fq'][i]) for i in range(len(result_save['Fq'])-5, len(result_save['Fq'])-1)]
                if max(diffs) <= 1e-3 or Fq.item() >= 0.99:
                    stop_i = i
                    break

        if (not curvature_too_large and satisfied_step) or satisfied_fval or condchange < opts['mincondchange']:
            stop_i = i
            break

    pbar.close()

    if stop_i == -1:
        stop_i = epochs - 1
        
    return rho, stop_i, time_all


if __name__ == '__main__':
    pass
