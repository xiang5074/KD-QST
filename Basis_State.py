# -*- coding: utf-8 -*-
# @Author: Xiang Li, foxwy
# @Date:   2025-12-01 
# @Paper 1:  Efficient quantum state tomography with two complementary measurements
# @Paper 2:  Direct Quantum State Tomography Based on Kirkwood-Dirac Quasiprobability

# Quantum state helpers for generating canonical and random multi-qubit states


import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import time
sys.path.append('..')


data_type = torch.complex128

class State():
    """
    Some basic quantum states and matrices, including |0>, |1>, Pauli matrices, GHZ_P,
    GHZi_P, Product_P, W_P, and some random states.

    Examples::
        >>> st = State()
        >>> GHZ_state = st.Get_GHZ_P(1, 0.3)
        >>> (matrix([[0.70710678], [0.70710678]]), 
              array([[0.5 , 0.15], 
                     [0.15, 0.5 ]]))
        >>> GHZ_state = st.Get_state_rho('GHZ_P', 1, 0.3)
    """
    def __init__(self):
        '''# Pauli matrices
        self.I = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64)
        self.X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        self.s1 = self.X
        self.Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        self.s2 = self.Y
        self.Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        self.s3 = self.Z'''

        # state
        self.state0 = torch.tensor([[1], [0]], dtype=data_type)
        self.state1 = torch.tensor([[0], [1]], dtype=data_type)
        self.state01 = 1 / torch.sqrt(torch.tensor(2.0, dtype=data_type)) * (self.state0 + self.state1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def Get_state_rho(self, N, state_name, p=1, purity=1, rank=1):
        """
        Obtain the corresponding quantum state based on the input.

        Args:
            state_name (str): The name of quantum states, include [GHZ_P], [GHZi_P],
                [Product_P], [W_P], [random_P], [real_random].
            N (int): The number of qubits.
            p (int): The P of Werner state, pure state when p == 1, identity matrix when p == 0.

        Returns:
            matrix: Pure state.
            array: Rho, mixed state.

        Examples:
            >>> st = State()
            >>> GHZ_state = st.Get_state_rho('GHZ_P', 1, 0.3)
            >>> (matrix([[0.70710678], [0.70710678]]), 
                  array([[0.5 , 0.15], 
                         [0.15, 0.5 ]]))
        """
        if state_name == 'GHZ':
            state, rho = self.Get_GHZ_P(N, p)
        elif state_name == 'W':
            state, rho = self.Get_W_P(N, p)
        elif state_name == 'Werner':
            state, rho = self.Get_Werner_p(N, p)        
        elif state_name == 'random_Purity':
            state, rho = self.Get_random_state_purity(N, purity)
        elif state_name == 'random_Rank':
            state, rho = self.Get_random_state_rank(N, rank)
        else:
            print('sorry, we have not yet achieved other quantum states!!!')
        return state, rho

    def Get_GHZ_P(self, N, p):
        """
        N_qubit Werner state with GHZ state,
        rho = p * |GHZ><GHZ| + (1 - P) / d * I.

        Args:
            N (int): The number of qubits.
            p (float): 0 <= p <= 1, |GHZ> when p == 1, I when p == 0.

        Returns:
            tensor: |GHZ>.
            tensor: rho.
        """
        assert 0 <= p <= 1, 'please input ``p`` of [0, 1]'

        state0 = self.state0
        state1 = self.state1
        for _ in range(N - 1):
            state0 = torch.kron(state0, self.state0)
            state1 = torch.kron(state1, self.state1)
        GHZ_state = 1 / torch.sqrt(torch.tensor(2.0)) * (state0 + state1)
        GHZ_rho = GHZ_state @ GHZ_state.T.conj()

        # GHZ_P
        GHZ_P_rho = p * GHZ_rho + (1 - p) / 2**N * torch.eye(2**N, dtype=data_type)
        return GHZ_state, GHZ_P_rho

    def Get_W_P(self, N, p):
        """
        N_qubit Werner state with W state,
        rho = p * |W><W| + (1 - P) / d * I.

        Args:
            N (int): The number of qubits.
            p (float): 0 <= p <= 1, |W> when p == 1, I when p == 0.

        Returns:
            tensor: |W>.
            tensor: rho.
        """
        assert 0 <= p <= 1, 'please input ``p`` of [0, 1]'

        I_array = torch.eye(N, dtype=torch.int32)
        W_state = torch.zeros((2**N, 1), dtype=data_type)
        for row in I_array:
            W_state += self.Get_state_from_array(row)
        W_state = 1 / torch.sqrt(torch.tensor(N, dtype=torch.float64)) * W_state
        W_rho = W_state @ W_state.T.conj()

        # W_P
        W_P_rho = p * W_rho + (1 - p) / 2**N * torch.eye(2**N, dtype=data_type)
        return W_state, W_P_rho

    def Get_state_from_array(self, array):
        """Calculate the corresponding pure state according to the given array"""
        st = {0: self.state0, 1: self.state1}
        State = st[array[0].item()]
        for i in array[1:]:
            State = torch.kron(State, st[i.item()])
        return State
    
    def Get_random_state_purity(self, N, purity):
        """
        Generate ``N``-qubit mixed states of corresponding ``purity`` with exponential 
        decay of eigenvalues, see paper ``Projected gradient descent algorithms for 
        quantum state tomography``.

        Args:
            N (int): The number of qubits.
            purity (float): The purity of mixed states, 0 <= purity <= 1.

        Returns:
            rho, if purity != 1, otherwise, pure state.
            rho, mixed state.
        """
        assert 0 <= purity <= 1, 'Please input purity of [0, 1]'

        if purity != 1:
            lambda_t = 0
            purity_t = 0
            x = torch.arange(1, 2**N + 1, dtype=torch.float32)

            while purity_t < purity:
                lambda_t += 0.001
                lam = torch.exp(-lambda_t * x)
                lamb = lam / torch.sum(lam)
                purity_t = torch.sum(lamb ** 2).item()

            randM = torch.rand((2**N, 2**N), dtype=torch.float32) * \
                     torch.exp(1j * 2 * torch.pi * torch.rand((2**N, 2**N), dtype=torch.float32))

            # QR decomposition
            Q, _ = torch.linalg.qr(randM)
            lamb_tmp = torch.abs(lamb) / torch.sum(torch.abs(lamb))
            rho = (Q * lamb_tmp.view(-1, 1)).matmul(Q.T.conj())
            #rho = self.proj_spectrahedron(rho)
            return rho, rho
        else:
            x_r = torch.rand((2**N, 1), dtype=torch.float32) * 2 - 1  # uniform(-1, 1)
            x_i = torch.rand((2**N, 1), dtype=torch.float32) * 2 - 1  # uniform(-1, 1)
            x = x_r + 1j * x_i
            x /= torch.norm(x)

            rho = x.matmul(x.T.conj())
            return x, rho

    def Get_random_state_rank(self, N, rank):
        rho_T = torch.randn((2**N, rank), dtype=torch.float32, device=self.device) + \
            1j * torch.randn((2**N, rank), dtype=torch.float32, device=self.device)
        #rho_T /= torch.norm(rho_T)
        state = rho_T
        rho_T = rho_T @ rho_T.T.conj()
        rho_T /= torch.trace(rho_T)
 
        return state, rho_T

    def Get_Werner_p(self, N, p=None):
        """
        Random Werner state
        """
        psi, rho = self.Get_random_state_rank(N, 1)

        if p is None:  # random
            p = torch.rand(1).item()
        else:  # given
            assert p >= 0 and p <= 1, print('please input ``p`` of [0, 1]')

        rho = p * rho + (1 - p) / 2**N * torch.eye(2**N).to(device=rho.device)
        return psi, rho


    def Is_rho(self, rho):
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
        elif not self.semidefinite_adjust(rho):
            print('rho is not positive semidefine')
            return 0
        else:
            return 1

    
    @staticmethod
    def semidefinite_adjust(M, eps=1e-08):
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
            return False



#--------------------main--------------------
if __name__ == '__main__':
    pass
