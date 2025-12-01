# -*- coding: utf-8 -*-
# @Author: Xiang Li, foxwy
# @Date:   2025-12-01 
# @Paper 1:  Efficient quantum state tomography with two complementary measurements
# @Paper 2:  Direct Quantum State Tomography Based on Kirkwood-Dirac Quasiprobability

# KDQ module implementing quasiprobability generation and quantum state tomography routines


import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import torch.nn.functional as F
import time
import numpy as np
from numpy.random import default_rng
from Basis_State import State
from Basis_Function import qmt_torch, qmt_matrix_torch, \
                           proj_spectrahedron, cal_Fidelity, \
                           shuffle_forward_torch, shuffle_adjoint_torch, shuffle_sample,\
                           monitor_memory, ten_to_k
from others.qse_apg import qse_apg
from others.LRE import LRE
if torch.cuda.is_available():
    torch.backends.cuda.preferred_linalg_library('magma')
                     
from circuit import Circuit_meas
N_moniter = 14
N_complex128 = 10

def softmax(x, dim=0):
    x_exp = torch.exp(x)
    sum_exp = torch.sum(x_exp, dim=dim, keepdim=True)
    softmax = x_exp / sum_exp
    return softmax

class KDQ:
    def __init__(self, args):
        self.N = args.N
        self.purity = args.purity
        self.sample_size = args.sample_size
        self.shots = args.shots
        self.n_epochs = args.n_epochs if args.rankflag==0 else 2000
        self.parallel_flag = args.parallel_flag
        self.exp_flag = args.exp_flag 
        self.state_flag = args.state_flag
        self.mea_flag = args.mea_flag
        self.error_flag = args.error_flag
        self.decom_flag = args.decom_flag
        self.rankflag = args.rankflag
        self.his_flag = args.his_flag
        self.noise_strength = args.noise_strength
        self.rank = args.rank
        self.device = args.device
        #print(self.device)
        #self.device = 'cpu'

        self.p = np.sqrt((self.purity - 1/2**self.N) / (1 - 1/2**self.N))      
        self.data_type = torch.complex128 if self.N <= N_complex128 else torch.complex64      


        # state initialization
        self.state, self.rho = State().Get_state_rho(self.N, self.state_flag, self.p, self.purity, self.rank)
        # Ensure state tensors live on the requested device
        self.state = self.state.to(device=self.device, dtype=self.data_type)
        self.rho = self.rho.to(device=self.device, dtype=self.data_type)
        
        # KD operator basis of single qubit
        if self.parallel_flag == 0:
            self.mea1_eig = torch.tensor([[1, 0], [0, 1]], dtype=self.data_type, device=self.device)
            self.mea2_eig = torch.tensor([[1, 1], [1, -1]], dtype=self.data_type, device=self.device) / torch.sqrt(torch.tensor(2.0))
            #self.mea2_eig = torch.tensor([[1, torch.sqrt(torch.tensor(3))], [torch.sqrt(torch.tensor(3)), -1]], dtype=self.data_type, device=self.device) / torch.tensor(2)
        elif self.parallel_flag == 1:
            self.mea1_eig = torch.tensor([[1, 0], [0, 1]], dtype=self.data_type, device=self.device)
            self.mea2_eig = torch.tensor([[1, 0], [0, 1]], dtype=self.data_type, device=self.device)


    def cir_pef(self):  # determine q_ij under different experimental conditions
        M,_ = self.cal_M() 

        if self.exp_flag == 0:  # the direct calculation of qij using formula                       
            self.rho = self.rho.to(self.device)
            q_ij = qmt_torch(self.rho, [M]*self.N)
            self.rho = self.rho.to('cpu')          # release memory after using self.rho
            if self.mea_flag == 'kd':       # KD case, add Gaussian noise to both real and imag parts
                q_ij.real = q_ij.real + torch.normal(mean=0., std=self.noise_strength / 2**self.N, size=q_ij.real.shape, device=self.device)   
                q_ij.imag = q_ij.imag + torch.normal(mean=0., std=self.noise_strength / 2**self.N, size=q_ij.imag.shape, device=self.device)   
                q_ij = q_ij.real + 1j * q_ij.imag
            else:     # Pauli case, q_ij are real probabilities
                q_ij = q_ij + torch.normal(mean=0., std=self.noise_strength / 2**self.N, size=q_ij.real.shape, device=self.device)
            
        elif self.exp_flag == 1:   # qiskit circuit simulation to obtain qij
            rho_eye = torch.eye(2**self.N, dtype=self.data_type, device=self.device) / 2**self.N
            probas = qmt_torch(rho_eye, [M]*self.N)
            
            circ_sim = Circuit_meas(state=self.state, N=self.N, p=self.p, probas=probas, parallel_flag=self.parallel_flag, error_flag=self.error_flag, decom_flag=self.decom_flag, noise_strength = self.noise_strength, shots=self.shots, backend='aer')
            q_ij = circ_sim.simulation() #.reshape(-1,1)

            # Reorder qij to match the layout produced by qmt_torch
            Ds = torch.tensor([2]*self.N, dtype=torch.int)   
            q_ij = shuffle_forward_torch(q_ij.T, Ds)
            q_ij = q_ij.reshape(-1, 1).to(device=self.device)
        
        elif self.exp_flag == 2:     # the simulation of experimental measurement process to obtain qij from given number of samples
            if self.mea_flag == 'kd':      # simulate kd measurement process to obtain qij           
                p_all = torch.zeros((2**self.N, 2**self.N, 4), dtype=torch.float64, device=self.device)
                q_all = torch.zeros((2**self.N, 2**(self.N+1), 2), dtype=self.data_type, device=self.device)
                q_ij = torch.zeros((2**self.N, 2**self.N), dtype=self.data_type, device=self.device)
           
                q_ij_t = qmt_torch(self.rho, [M]*self.N)    # calculate ideal q_ij=Tr(\rho\Pi_i\Pi_j)       
                # Reorder qij from qmt_torch back to the natural order
                Ds = torch.tensor([2]*self.N, dtype=torch.int) 
                q_ij_t = shuffle_adjoint_torch(q_ij_t, Ds)

                # calculate ideal Tr(\rho\Pi_j)
                Mj = torch.zeros((len(self.mea2_eig), 2, 2), dtype=self.data_type, device=self.device)
                for j in range(len(self.mea2_eig)):
                    psi_2 = self.mea2_eig[j, :].reshape(-1, 1)
                    Mj[j] = psi_2 @ psi_2.conj().T
                term1_m = qmt_torch(self.rho, [Mj]*self.N)

                # calculate ideal Tr(\Pi_i\Pi_j)
                identity_m = torch.eye(2**self.N, dtype=self.data_type, device=self.device)
                term2_m = qmt_torch(identity_m, [M]*self.N)
                term2_m = shuffle_adjoint_torch(term2_m, Ds)    # Reorder term2_m into the natural layout              


                for i in range(2**self.N):
                    for j in range(2**self.N):
                        qij = q_ij_t[i, j]
                        for c in range(4):
                            term1 = term1_m[j]
                            term2 = term2_m[i, j]
                            h = self.h_step(1 - c)
                            h2 = self.h_step(c - 2)
                            p_all[i, j, c] = 1 / 4 *  (term1 + term2 + 2 * h * (-1)**c * qij.real + 2 * h2 * (-1)**c * qij.imag).real       

                shots = int(np.ceil(self.shots / 2**self.N))
                for i in range(2**self.N):                    
                    # Sample from two multinomial distributions to mimic statistical noise
                    q_all[i,:,0] = self.sample(shots, torch.hstack((p_all[i,:,0].cpu(), p_all[i,:,1].cpu())))  # X                
                    q_all[i,:,1] = self.sample(shots, torch.hstack((p_all[i,:,2].cpu(), p_all[i,:,3].cpu())))  # Y
                    q_ij[i, :] = (q_all[i, 0:2**self.N, 0] - q_all[i, 2**self.N:, 0]) + 1j * (q_all[i, 0:2**self.N, 1] - q_all[i, 2**self.N:, 1])

                qij_re = q_ij.real
                qij_re /= torch.sum(qij_re)  
                qij_im = q_ij.imag
                qij_im = qij_im - torch.sum(qij_im) / qij_im.numel()
                q_ij = qij_re + 1j * qij_im
                #print('q_ij:', q_ij)

                # Reorder qij to match qmt_torch output order
                q_ij = shuffle_forward_torch(q_ij.T, Ds)
                q_ij = q_ij.reshape(-1, 1)
            
            else:      # simulate pauli measurement process to obtain qij
                MP = {}   # Z,X,Y
                MP[0] = torch.stack([M[0], M[1]])
                MP[1] = torch.stack([M[2], M[3]])
                MP[2] = torch.stack([M[4], M[5]])
                
                p_all = torch.zeros((3**self.N, 2**self.N), dtype=self.data_type, device=self.device)
                shots = int(self.shots / 3**self.N)
                for i in range(p_all.shape[0]):     
                    index = ten_to_k(i, 3, self.N)   # convert from decimal to ternary digits
                    MP_i = [MP[index[j]] for j in range(self.N)]
                    p_all[i] = qmt_torch(self.rho, MP_i).real.flatten() * 3**self.N
                    p_all[i] = self.sample(shots, p_all[i])      

                p_all = p_all / 3**self.N
                #print('p_all_0:', p_all)
                q_ij = shuffle_sample(p_all)   # Reorder samples to align with qmt_torch output
                
        
        if self.parallel_flag == 0:     # normalization of q_ij if not parallel measurement
            self.qij = q_ij / torch.sum(q_ij) 
        elif self.parallel_flag == 1:   # parallel case, donnot satisfy normalization
            self.qij = q_ij
        #print(self.qij)


    @staticmethod
    def h_step(t):
        return 1 if t >= 0 else 0

    def sample(self, Ns, P_data):
        rng = default_rng()
        P_data = P_data.cpu().numpy().astype(np.double)
        counts = rng.multinomial(Ns, P_data / sum(P_data))
        counts = torch.from_numpy(counts)
        P_sample = (counts / Ns).to(dtype=self.data_type, device=self.device)
        return P_sample

    def cal_M(self):
        if self.mea_flag == 'kd':
            K = len(self.mea1_eig) * len(self.mea2_eig)
            M = torch.zeros((K, 2, 2), dtype=self.data_type, device=self.device)
            MPDE = torch.zeros((K, 2, 2), dtype=self.data_type, device=self.device)
            for i in range(len(self.mea1_eig)):
                for j in range(len(self.mea2_eig)):
                    psi_1 = self.mea1_eig[i, :].reshape(-1, 1)
                    mea1 = psi_1 @ psi_1.conj().T
                    psi_2 = self.mea2_eig[j, :].reshape(-1, 1)
                    mea2 = psi_2 @ psi_2.conj().T
                    add_state = torch.ones((2,2), dtype=self.data_type, device=self.device) / 2
                    m = mea1 @ mea2 if self.parallel_flag==0 else mea1 @ add_state @ mea2      # parallel measurement or not
                    M[i * len(self.mea2_eig) + j] = m                        # unnormalized trace
                    MPDE[i * len(self.mea2_eig) + j] = m / torch.trace(m)    # trace normalized
        elif self.mea_flag == 'Pauli_POVM':   # Z,Y,X
            K = 6
            Ps = torch.tensor([1. / 3., 1. / 3., 1. / 3.], dtype=self.data_type, device=self.device)
            M = torch.zeros((K, 2, 2), dtype=self.data_type, device=self.device)
            M[0] = Ps[0] * torch.tensor([[1, 0], [0, 0]], dtype=self.data_type, device=self.device)
            M[1] = Ps[0] * torch.tensor([[0, 0], [0, 1]], dtype=self.data_type, device=self.device)
            M[4] = Ps[1] / 2 * torch.tensor([[1, 1], [1, 1]], dtype=self.data_type, device=self.device)
            M[5] = Ps[1] / 2 * torch.tensor([[1, -1], [-1, 1]], dtype=self.data_type, device=self.device)
            M[2] = Ps[2] / 2 * torch.tensor([[1, -1j], [1j, 1]], dtype=self.data_type, device=self.device) 
            M[3] = Ps[2] / 2 * torch.tensor([[1, 1j], [-1j, 1]], dtype=self.data_type, device=self.device)   
            MPDE = M

        elif self.mea_flag == 'Pauli_normal':  # projective measurement, not POVM
            K = 4
            M = torch.zeros((K, 2, 2), dtype=self.data_type, device=self.device)
            M[0] = 1 / torch.sqrt(torch.tensor(2.0)) * torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=self.data_type, device=self.device)
            M[1] = 1 / torch.sqrt(torch.tensor(2.0)) * torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=self.data_type, device=self.device)
            M[2] = 1 / torch.sqrt(torch.tensor(2.0)) * torch.tensor([[0.0, -1j], [1j, 0.0]], dtype=self.data_type, device=self.device)
            M[3] = 1 / torch.sqrt(torch.tensor(2.0)) * torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=self.data_type, device=self.device)
            MPDE = M

        elif self.mea_flag == 'Tetra4':
            self.I = np.array([[1, 0], [0, 1]])
            self.X = np.array([[0, 1], [1, 0]])
            self.Y = np.array([[0, -1j], [1j, 0]])
            self.Z = np.array([[1, 0], [0, -1]])
            M = np.zeros((4, 2, 2), dtype=np.complex128)
            M[0, :, :] = 0.25 * (self.I + (self.X + self.Y + self.Z) / np.sqrt(3))
            M[1, :, :] = 0.25 * (self.I + (-self.X - self.Y + self.Z) / np.sqrt(3))
            M[2, :, :] = 0.25 * (self.I + (-self.X + self.Y - self.Z) / np.sqrt(3))
            M[3, :, :] = 0.25 * (self.I + (self.X - self.Y - self.Z) / np.sqrt(3))
            M = torch.tensor(M, dtype=self.data_type, device=self.device)
            MPDE = M
        else:
            print(self.basis, 'does not exist!')

        return M, MPDE
    

    # ---------------------------------do tomography methods------------------------------------------------

    def QST_LIE(self):     # Two-step direct reconstruction method           
        if self.parallel_flag == 0:
            start_time = time.time()
            _, M_PDE = self.cal_M()
            rho = qmt_matrix_torch(self.qij.reshape(1,-1).conj(), [M_PDE]*self.N)   # reshape keeps the flattened element order
        elif self.parallel_flag == 1:
            # adjoint shuffle to get rho from qij
            Ds = torch.tensor([2]*self.N, dtype=torch.int)   
            qij = shuffle_adjoint_torch(self.qij, Ds)
            start_time = time.time()
            rho = torch.sqrt(torch.tensor(2))**(2*self.N) * qij.T
            print('trace rho before proj:', torch.trace(rho))          

        if self.N >= N_moniter:
            if self.N>=15:
                self.qij.to('cpu')       # release memory
            print('---qmt---')
            monitor_memory()
            t1 = time.time()
            #t_qmt = t1 - start_time
            #print('t_qmt:', t_qmt)
            rho = rho.cpu()

        # Projection to the nearest physical density matrix
        rho = proj_spectrahedron(rho)

        if self.N >= N_moniter:
            t_proj = time.time() - t1
            print('\n t_proj:', t_proj)

        t = time.time() - start_time

        if self.N >= N_moniter:
            print('---proj---')
            monitor_memory()
        return rho, t
    

    def QST_MLE(self):        # Gradient descent with maximum likelihood objective function
        time_all = 0
        time_b = time.time() 

        # Initial estimation
        A = torch.randn(2**self.N, 2**self.N).to(dtype=self.data_type, device=self.device)
        A = A @ A.conj().T
        rho = A / torch.trace(A)
        del A

        data = self.qij.reshape(-1, 1).to(device=self.device, dtype=torch.complex128)  # mixed precision calculation
        if self.N >= N_moniter:
            del self.qij
            monitor_memory()

        if self.mea_flag == 'kd' or self.mea_flag == 'Pauli_normal':
            if self.mea_flag == 'kd':
                # Data softmax
                data = torch.vstack((data.real, data.imag)) 
                if self.N >= N_moniter:
                    monitor_memory()
            #data = F.softmax(data, dim=0)
            data =  F.softmax(data.real, dim=0) if self.N<=14 else softmax(data.real, dim=0)
        if self.N >= N_moniter:
            monitor_memory()

        # Setting learning rate
        gam = 2 * 7.9**self.N if self.parallel_flag == 0 else 3.5 * 13**self.N   # MUB basis; for parallel case, step size needs further tuning
        #gam = 3 * 5.5**self.N  #2.5 * 5.8**self.N if self.N<=5 else 3 * 5.5**self.N   # non-MUB basis
        gam = torch.tensor(gam, device=self.device)

        M,_ = self.cal_M()
        probas = self._cost(rho, M)

        time_e = time.time() 
        time_all += time_e - time_b
        
        fidelity_his = []     
        fidelity = cal_Fidelity(rho, self.rho).item()
        fidelity_his.append(fidelity)  

        for k in range(self.n_epochs):   # (5):            
            time_b = time.time() 
            # Calculate gradient
            g = -(data[0:int(len(data)/2)]  - probas[0:int(len(probas)/2)]).to(self.data_type)\
                    - 1j * (data[int(len(data)/2):] - probas[int(len(probas)/2):]).to(self.data_type)
            if self.N >= N_moniter:
                print('k=',k)
                del probas                                                           
                monitor_memory()     
            G = qmt_matrix_torch(g.conj(), [M]*self.N)  
            if self.N >= N_moniter:
                del g
                monitor_memory()

            # Update rho
            rho -= gam * G
            if self.N >= N_moniter:
                monitor_memory()
                del G
                monitor_memory()   # use moniter_memory twice to ensure the release of memory
                rho = rho.cpu()
            rho = proj_spectrahedron(rho)
            
            # Calculate probabilities
            probas = self._cost(rho, M) 

            time_e = time.time() 
            time_all += time_e - time_b
            if self.N <= 12:
                # save fidelity
                fidelity = cal_Fidelity(rho, self.rho).item()
                fidelity_his.append(fidelity)
                        
                if k >= 5 and self.his_flag==0:
                    diffs = [abs(fidelity_his[i+1] - fidelity_his[i]) for i in range(len(fidelity_his)-6, len(fidelity_his)-1)]
                    if max(diffs) < 1e-5 or fidelity > 0.9999: #or k>=50:
                        break
            elif k >= 8:
                break

        if self.N >= N_moniter:
            del data, probas
            monitor_memory()

        print('MLE k:', k)
        if self.his_flag == 0:
            return rho, time_all
        else:
            return fidelity_his


    def _cost(self, rho, M):
        #rho = rho.to(dtype=torch.complex128)
        #M = M.to(dtype=torch.complex128)
        probas = qmt_torch(rho, [M]*self.N)
        if self.parallel_flag == 0:
            probas = probas / torch.sum(probas)  # q_ij normalization
        probas = probas.to(dtype=torch.complex128)    # mixed precision calculation
                
        if self.mea_flag == 'kd' or self.mea_flag == 'Pauli_normal':
            if self.mea_flag == 'kd':
                probas = torch.vstack((probas.real, probas.imag)) 
                if self.N >= N_moniter:
                    monitor_memory()
            #probas = F.softmax(probas.real, dim=0)
            probas = F.softmax(probas.real, dim=0) if self.N<=14 else softmax(probas.real, dim=0)
        if self.N >= N_moniter:
            monitor_memory()
        return probas

    def QST_CGAPG(self):
        result_save = {'rho': [],
                       'time': [], 
                       'epoch': [],
                       'Fq': []}

        self.mea_flag = 'Pauli_POVM'
        self.cir_pef()
        data_all = self.qij
        M,_ = self.cal_M()

        if self.N >= 10:   # reduce GPU pressure for larger systems
            data_all = data_all.to('cpu')
            self.rho = self.rho.to('cpu')
            monitor_memory()

        qse_apg(M, self.N, data_all, self.n_epochs, self.his_flag, result_save, device=self.device, rho_star=self.rho, data_type=self.data_type)

        if self.his_flag == 0:
            return result_save['rho'], result_save['time']
        else:
            return result_save['Fq']


    def QST_LRE(self):
        self.mea_flag = 'Pauli_POVM'
        self.cir_pef()
        data_all = self.qij
        M,_ = self.cal_M()       
        self.mea_flag == 'Pauli_normal'
        M_basis,_ = self.cal_M()
        rho, t = LRE(M, self.N, data_all, M_basis, device=self.device, data_type=self.data_type)

        return rho, t
    


if __name__ == '__main__':
    pass
