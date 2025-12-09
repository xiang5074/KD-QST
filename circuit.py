# -*- coding: utf-8 -*-
# @Author: Xiang Li, foxwy
# @Date:   2025-12-01 
# @Paper 1:  Efficient quantum state tomography with two complementary measurements
# @Paper 2:  Direct Quantum State Tomography Based on Kirkwood-Dirac Quasiprobability

# Qiskit-based circuit builder for KDQ measurement simulation with noise modeling


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import matplotlib.pyplot as plt

# Qiskit module
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, partial_trace
from qiskit.circuit.library import TGate
from qiskit_aer import AerSimulator  #, Aer

# Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel,
                              QuantumError,
                              ReadoutError,
                              depolarizing_error,
                              pauli_error,
                              thermal_relaxation_error)
from Basis_State import State
from Basis_Function import cal_Fidelity, ten_to_k


root_path = os.path.join(os.path.dirname(__file__), "results")

class Circuit_meas():
    """
    Implementation of all qiskit simulator parts as a source of
    experimental data for fidelity estimation.
    """

    def __init__(self, state=None, N=1, p=1, shots=4096, probas=0, noise_strength=0, error_flag='gatenoise', decom_flag=1, backend="aer", circtype='Random'):
        """
        Args:
            circtype (str): Type of state of the experiment, choices =
                ["GHZ", "W", "Random_Haar"].
            n_qubits (int): The number of qubits.
            backend (str): Qiskit experiment backend, choices =
                ['aer', 'qasm', 'FakePerth', 'MPS', 'stabilizer', 'IBMQ'].
            p (float): Level of werget state
        """
        assert circtype in ("GHZ", "W", "Random"), print(
            'please input right state type')
        assert backend in ('aer', 'qasm', 'MPS', 'stabilizer', 'IBMQ'), print(
            'please input right backend')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #'cpu' 
        self.state = state #/ torch.norm(state)
        self.rho = state.matmul(state.T.conj())
        self.N = N              # the number of qubits
        self.n_qubits = 2*N + 1 # the number of qubits used in circuit
        self.p = torch.tensor(p, device=self.device)              # Werner state weight
        self.probas = probas
        self.shots = int(shots/2**self.N)  # number of shots      
        self.noise_strength = noise_strength
        self.circtype = circtype  # the type of state        
        self.decom_flag = decom_flag
        self.error_flag = error_flag

        self.backend = self.get_backend(backend)        


    def get_backend(self, backend):
        """The backend of quantum machine"""

        if backend == 'aer':   
            noise_model = NoiseModel()
            one_qubit_gates = ['h', 'x', 'y', 'z', 's', 'sdg']

            # -------- Single-qubit gates: compose all configured errors once --------
            oneq_error = None

            if 'depolarizing' in self.error_flag:
                # Single-qubit depolarizing
                prob_1 = 0.001  # Baseline single-qubit gate error
                dep1 = depolarizing_error(prob_1, 1)
                oneq_error = dep1 if oneq_error is None else oneq_error.compose(dep1)

            if 'bitflip' in self.error_flag:
                p_bf = self.noise_strength
                bf = pauli_error([('X', p_bf), ('I', 1 - p_bf)])
                oneq_error = bf if oneq_error is None else oneq_error.compose(bf)

            if 'phaseflip' in self.error_flag:
                p_pf = self.noise_strength
                pf = pauli_error([('Z', p_pf), ('I', 1 - p_pf)])
                oneq_error = pf if oneq_error is None else oneq_error.compose(pf)

            if oneq_error is not None:
                noise_model.add_all_qubit_quantum_error(oneq_error, one_qubit_gates)

            # -------- Multi-qubit gates: compose once and apply --------
            if 'depolarizing' in self.error_flag:
                # Default multi-qubit gate noise strength
                prob_multi = 1e-5  # Use 1e-5 as the floor when noise is set to zero
                if self.decom_flag:
                    twoq_error = depolarizing_error(prob_multi, 2)
                    noise_model.add_all_qubit_quantum_error(twoq_error, ['cx'])
                else:
                    threeq_error = depolarizing_error(prob_multi, 3)
                    noise_model.add_all_qubit_quantum_error(threeq_error, ['cswap'])

            # -------- Readout error: independent from gate errors --------
            if 'readouterror' in self.error_flag:
                p01 = 1e-5
                p10 = p01
                readout_err = ReadoutError([[1 - p10, p10], [p01, 1 - p01]])
                noise_model.add_all_qubit_readout_error(readout_err)
            # Get basis gates from noise model
            basis_gates = noise_model.basis_gates

            backend = AerSimulator(noise_model=noise_model, basis_gates=basis_gates)
            #backend = Aer.get_backend('aer_simulator')  # aer_simulator
        #elif backend == 'MPS':
        #    backend = Aer.get_backend('aer_simulator_matrix_product_state')
        elif backend == 'stabilizer':
            backend = AerSimulator(method='extended_stabilizer')
        else:
            print('please input right backend')

        return backend


    # ----------------initialization----------------
    def get_init_circuit(self, circtype=None, eigenstate_target=None, eigenstate_swap=None):
        if circtype is not None:
            self.circtype = circtype

        qr = QuantumRegister(self.n_qubits)
        cr = ClassicalRegister(self.n_qubits)
        qc = QuantumCircuit(qr, cr, name='st_circ')
        self.qr = qr
        self.qc = qc
        # initialization of the target system
        self.init_target(eigenstate_target)

        # initialization of the swap system
        eigenstates = eigenstate_swap + ['+']
        self.init_ancilla(eigenstates)

        '''# Obtain the state prepared with gate noise
        rho_noise = DensityMatrix(self.qc)
        rho_noise = partial_trace(rho_noise, list(range(self.N, self.n_qubits)))
        rho_noise = rho_noise.reverse_qargs().data
        fq = cal_Fidelity(self.rho.cpu(), rho_noise)  # Not accurate because rho_noise ordering differs
        return fq'''


    def init_target(self, eigenstate_target=None):
        """prepare desired quantum state"""

        # obtain the circuit according to the type of state
        if self.circtype == "GHZ":  # GHZ-class state
            choice = [1.0]  # , -1.0, 1.0j, -1.0j]
            w_s = np.random.choice(choice, 1, replace=True)[0]
            self.w_s = w_s

            if w_s == 1:  # GHZ
                self.qc.h(0)
                for i in range(1, self.N):
                    self.qc.cx(0, i)
            elif w_s == -1:
                self.qc.x(0)
                self.qc.h(0)
                for i in range(1, self.N):
                    self.qc.cx(0, i)
            elif w_s == 1.0j:
                self.qc.h(0)
                self.qc.s(0)
                for i in range(1, self.N):
                    self.qc.cx(0, i)
            elif w_s == -1.0j:
                self.qc.x(0)
                self.qc.h(0)
                self.qc.s(0)
                for i in range(1, self.N):
                    self.qc.cx(0, i)

        elif self.circtype == "W":  # W-class state
            theta = np.arccos(1 / np.sqrt(self.N))
            self.qc.ry(2 * theta, 0)
            for i in range(self.N - 2):
                theta = np.arccos(1 / np.sqrt(self.N - i - 1))
                self.qc.cry(2 * theta, i, i + 1)
            for i in range(self.N - 1):
                self.qc.cx(self.N - 2 - i, self.N - 1 - i)
            self.qc.x(0)

            # [1, -1, 1j, -1j]
            choice = [1.0]  # , -1.0, 1.0j, -1.0j]
            w_s = np.random.choice(choice, self.N, replace=True)
            self.w_s = w_s
            for i in range(self.N):
                if w_s[i] == -1:
                    self.qc.z(i)
                elif w_s[i] == 1j:
                    self.qc.s(i)
                elif w_s[i] == -1j:
                    self.qc.sdg(i)

        elif self.circtype == "Random":  # random state based on Haar measure
            eigenstate_target = [complex(x) for x in eigenstate_target.cpu().numpy().flatten()]
            eigenstate_target = eigenstate_target / np.linalg.norm(eigenstate_target)
            self.qc.initialize(eigenstate_target, range(self.N-1,-1,-1))   # Note the reversed qubit order

        else:
            print('please input right state type')



    def init_ancilla(self, eigenstates):
        anc_qc = self.prepare_specific_pauli_eigenstate(eigenstates=eigenstates)
        self.qc.compose(anc_qc, range(self.n_qubits), inplace=True)   



    def prepare_specific_pauli_eigenstate(self, eigenstates):
        """
        Prepare a specific eigenstate of a multi-qubit Pauli operator.

        Args:
            eigenstates (list): Per-qubit eigenstate labels, e.g., ['+', '1'] means
                |+> for X and |1> for Z.

        Returns:
            QuantumCircuit: Circuit that prepares the desired Pauli eigenstate.
        """
        
        qc = QuantumCircuit(self.n_qubits)
        
        # Apply single-qubit preparations on the ancilla register
        for idx, eigenstate in enumerate(eigenstates):
            i = self.N + idx
            if eigenstate == '+':
                qc.h(i)  # Prepare |+>
            elif eigenstate == '-':
                qc.x(i)  # Flip to |1>
                qc.h(i)  # Prepare |->
            elif eigenstate == 'i':
                qc.h(i)
                qc.sdg(i)  # Prepare (|0> + i|1>)/√2
            elif eigenstate == '-i':
                qc.x(i)  # Flip to |1>
                qc.h(i)
                qc.sdg(i)  # Prepare (|0> - i|1>)/√2
            elif eigenstate == '0':
                pass  # Leave as |0>
            elif eigenstate == '1':
                qc.x(i)  # Prepare |1>
        
        return qc

    # ----------------cswap----------------
    def cswap_circuit(self):
        cswap_qr = QuantumRegister(self.n_qubits, 'q')
        cswap_qc = QuantumCircuit(cswap_qr)       

        if self.decom_flag == 0:
            for i in range(self.N):
                cswap_qc.cswap(-1, i, self.N+i)
        elif self.decom_flag == 1:
            for i in range(self.N):
                self.cswapdecompose(cswap_qc, i, self.N+i, -1)

        self.qc.barrier()
        self.qc.compose(cswap_qc, range(self.n_qubits), inplace=True)        

    @staticmethod
    def cswapdecompose(qc,a,b,c):
        #qc=QuantumCircuit(3)
        qc.cx(a,b)
        qc.h(a)
        qc.cx(b,a)
        qc.append(TGate().inverse(), [a])
        qc.cx(c,a)
        qc.t(a)
        qc.cx(b,a)
        qc.append(TGate().inverse(), [a])
        qc.append(TGate().inverse(), [b])
        qc.cx(c,a)
        qc.cx(c,b)
        qc.t(a)
        qc.append(TGate().inverse(), [b])
        qc.h(a)
        qc.cx(c,b)
        qc.t(c)
        qc.s(b)
        qc.cx(a,b)
        return qc
    
    # ----------------measurement circuit----------------
    def get_mea_circuit_basis(self, meas_element):
        """
        quantum measurement based on different basis: Z:0, X:1, Y:2.

        Args:
            meas_element (list): Pauli measurement in each qubit, based on Qiskit, from bottom to top.
                example-["X", "Z", "Y"] for 3-qubit.

        Returns:
            Quantum circuit of measurement.
        """

        mea_qr = QuantumRegister(self.n_qubits, 'q')
        mea_qc = QuantumCircuit(mea_qr)

        for idx, m in enumerate(meas_element):
            if m in (1, "X"):  # X
                mea_qc.h(self.n_qubits - 1 - idx)
            elif m in (2, "Y"):  # Y
                mea_qc.sdg(self.n_qubits - 1 - idx)
                mea_qc.h(self.n_qubits - 1 - idx)

        return mea_qc

    def circuit_basis(self, meas_element):
        """
        Quantum circuit combines state circuit and measurement circuit.

        Args:
            meas_element (list): Pauli measurement in each qubit, based on Qiskit, from bottom to top.
                example-["X", "Z", "Y"] for 3-qubit.

        Returns:
            Final Quantum circuit.
        """

        mea_qc = self.get_mea_circuit_basis(meas_element)
        self.qc.barrier()
        self.qc.compose(mea_qc, range(self.n_qubits), inplace=True)
        self.qc.measure(list(range(self.N,self.n_qubits)),list(range(self.N,self.n_qubits)))

    

    # ----------------get measurement result----------------
    def get_measure(self):
        # Transpile the ideal circuit to a circuit that can be directly executed by the backend
        transpiled_circuit = transpile(self.qc, self.backend)

        # counts
        result = self.backend.run(transpiled_circuit, shots=int(self.shots)).result()
        counts = result.get_counts()
        #print('counts1', counts.items())

        counts = {key[:(self.N+1)][::-1]: value for key, value in counts.items()}  #if key[-1:self.N:-1] == '0'*self.N}  
                # Keep only measured qubits and reverse into natural order
                
        f = {key: value/int(self.shots) for key, value in counts.items()}  
        #print('f:', f)
        self.f = f
    


    # ----------------get qij----------------
    def data_process(self):
        f_obs = {}
        for i in range(2**(self.N)):
            i2 = ten_to_k(i, 2, self.N)
            key = "".join([str(j) for j in i2])
            f_add = self.f[key+'0'] if key+'0' in self.f.keys() else 0
            f_sub = self.f[key+'1'] if key+'1' in self.f.keys() else 0
            f_obs[key] = f_add - f_sub

        self.f_obs = f_obs
        return f_obs

    

    # ----------------run this----------------
    def simulation(self):
        eigen = ['+','-']

        k = 2**self.N 
        
        qij = torch.zeros((2**self.N, 2**self.N), dtype=torch.complex128, device=self.device)
        for j in range(k):    # Swap-system eigenstate determined by j
            j2 = ten_to_k(j, 2, self.N)
            eigenstate_swap = [eigen[j] for j in j2]

            q = {}
            for idx in [1,2]:  # 0:Z, 1:X, 2:Y
                self.get_init_circuit(eigenstate_target=self.state, eigenstate_swap=eigenstate_swap)
                self.cswap_circuit()   # cswap operation

                #pauli_string = [0]*self.N + [idx]
                self.circuit_basis([idx])
                
                # -------------------------Draw the circuit-------------------------
                '''self.qc.draw(output='mpl')
                # Save circuit diagram
                save_path = root_path #+ '/test.jpg'
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                    print('Result directory created: ' + save_path)
                else:
                    print('Result directory exists: ' + save_path)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()'''

                self.get_measure()
                q[idx] = self.data_process()   # Each q[idx] entry is a dictionary
                #print(f'q{idx}:{q[idx]}')


            qij_list = torch.tensor([q[1][key] - 1j * q[2][key] for key in sorted(q[1].keys())], dtype=torch.complex128)
            qij[:,j] = qij_list

        # Normalize so Re(qij) sums to 1 and Im(qij) sums to 0
        qij_re = qij.real
        qij_re /= torch.sum(qij_re)  
        qij_im = qij.imag
        qij_im = qij_im - torch.sum(qij_im) / qij_im.numel()
        qij = qij_re + 1j * qij_im    

        #final_qij = self.p * qij.reshape(-1,1) + (1 - self.p) * self.probas   # Uncomment to include Werner mixed state
        final_qij = qij.reshape(-1,1)   # Pure-state scenario only
    
        return final_qij #, fq




if __name__ == '__main__':
    N = 1
    shots = 8192  # Must be a power of two; 2^10 = 1024
    circtype = "Random"

    state_flag = 'GHZ'
    p = 0.6

    rho_np, _ = State().Get_state_rho(N, state_flag, p)
    rho = torch.tensor(rho_np, dtype=torch.complex128, device='cpu')

    circ_sim = Circuit_meas(state=rho, N=N, shots=shots, decom_flag=0, backend='aer')
    qij = circ_sim.simulation()
    print('qij:',qij)

    circ_sim = Circuit_meas(state=rho, N=N, shots=shots, decom_flag=0, backend='aer')
    qij = circ_sim.simulation()
    print('qij1:',qij)
    #print('sum qij:', torch.sum(qij))
