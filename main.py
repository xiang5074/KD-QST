"""
-----------------------------------------------------------------------------------------
    The main function of quantum state tomography, used in the experimental 
    part of the paper, calls other implementations of the QST algorithm,
    paper 1: ``# @Paper: Efficient quantum state tomography with two complementary measurements``
    paper 2: ``# @Paper: Direct Quantum State Tomography Based on Kirkwood-Dirac Quasiprobability``
    @Author: Xiang Li, foxwy
    @Date:   2025-12-01
-----------------------------------------------------------------------------------------
"""

# This code implements and compares various quantum state tomography methods:
# - Linear Inversion Estimation (LIE)                         ---- LIE with Kirkwood-Dirac quasiprobability
# - Maximum Likelihood Estimation (MLE)                       ---- MLE with Kirkwood-Dirac quasiprobability
# - Conjugate Gradient-Accelerated Projected Gradient (CGAPG) ---- MLE with Pauli probability
# - Linear-Regression Estimation (LRE)                        ---- LRE with Pauli probability
#
# The code evaluates these methods based on:
# - Fidelity vs. number of qubits
# - Runtime performance
# - Noise resilience
# - Sample complexity
# - Convergence behavior


import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import argparse
import traceback
from KDQ import KDQ
from Basis_Function import cal_Fidelity, monitor_memory
import numpy as np
import math
if torch.cuda.is_available():
    torch.backends.cuda.preferred_linalg_library('magma')     # Use Magma library for better performance on GPU
                     
from circuit import Circuit_meas
data_type = torch.complex128
root_path = os.path.join(os.path.dirname(__file__), "results")
N_moniter = 14



# Save results
def save_results(results, file_path, file_name):
    file_path = os.path.join(root_path, file_path.lstrip("/\\"))
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
        print('Result directory created: ' + file_path)
    else:
        print('Result directory exists: ' + file_path)
    np.save(os.path.join(file_path, file_name), results)

def save_file(file, paths, names):
    file_path = os.path.join(root_path, f"noise_{paths[0]}", str(paths[1]))
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
        print('Result directory created: ' + file_path)
    else:
        print('Result directory exists: ' + file_path)

    file_name = "_".join(str(name) for name in names) + ".npy"
    np.save(os.path.join(file_path, file_name), file)


# fidelity and runtime with the number of qubits of six algorithms
def exp_1(args):
    args.state_flag = 'random_Rank'   # {'random_Rank', 'Werner', 'W', 'GHZ'}
    args.mea_flag = 'kd'
    args.n_epochs = 500

    #with torch.amp.autocast('cuda'):
    for args.N in range(1,7):    
        noise_values = [1e-5]
        for args.noise_strength in noise_values:      
            methods = [1, 1, 1, 1]    # which methods to run: LIE, MLE, CGAPG(pauli), LRE(pauli)
            exp = 10                        # repetitions for each setting
            save_data = {
            'fq1': [],
            'time1': [],
            'fq2': [],
            'time2': [],
            'fq3': [],
            'time3': [],           
            'fq4': [],
            'time4': [],           
            }
            for _ in range(exp):
                if args.state_flag == 'random_Rank':
                    args.rank = torch.randint(1, 2**args.N + 1, (1,), device=args.device)[0].item()
                    #print(f'rank={args.rank}')
                elif args.state_flag == 'Werner':
                    purity_flag = 1
                    while purity_flag:
                        args.purity = torch.rand(1, device=args.device).item()
                        if args.purity >= 1/2**args.N:
                            purity_flag = 0
                elif args.state_flag == 'W' or 'GHZ':
                    args.purity = 1
                while True:
                    try:
                        # initialize KDQ object and perform measurement
                        start_time = time.time()
                        kdq = KDQ(args)  
                        if kdq.N >= N_moniter:
                            print('---init state---,time=',time.time()-start_time)
                            #monitor_memory()                                                                
                        kdq.cir_pef()
                        if kdq.N >= N_moniter:
                            print('---init qij---')
                            monitor_memory()
                            t = time.time() - start_time
                            print('t_initial:', t)

                        # Execute three quantum state tomography methods, save fidelity and time
                        if methods[0]:
                            rho1, t1 = kdq.QST_LIE()
                            if kdq.N >= N_moniter:
                                print('t1:',t1)   
                                if kdq.N >= 15:
                                    rho1 = rho1.to('cpu')
                                    kdq.rho = kdq.rho.to('cpu')
                                    monitor_memory()
                            
                            start_time = time.time()                                        
                            fq1_val = cal_Fidelity(rho1, kdq.rho).item() 
                            if kdq.N >= N_moniter:
                                t = time.time() - start_time #t_stamp
                                print('t_fid1:', t, ',fq1:', fq1_val)
                            
                            save_data['fq1'].append(fq1_val)
                            save_data['time1'].append(t1)   
                            del rho1 
                            monitor_memory()

                        if methods[1]:                        
                            rho2, t2 = kdq.QST_MLE()
                            #print('t2:',t2)
                            if kdq.N >= 15:
                                rho2 = rho2.to('cpu')
                                kdq.rho = kdq.rho.to('cpu')       

                            start_time = time.time()       
                            fq2_val = cal_Fidelity(rho2, kdq.rho).item()
                            if kdq.N >= N_moniter:
                                t = time.time() - start_time #t_stamp
                                print('t_fid2:', t, ',fq2:', fq2_val)

                            save_data['fq2'].append(fq2_val)
                            save_data['time2'].append(t2)   
                            del rho2
                            monitor_memory()

                        if methods[2]:                        
                            rho3, t3 = kdq.QST_CGAPG()
                            #print('t3:',t3)

                            start_time = time.time()       
                            fq3_val = cal_Fidelity(rho3, kdq.rho).item()
                            if kdq.N >= N_moniter:
                                t = time.time() - start_time #t_stamp
                                print('t_fid3:', t, ',fq3:', fq3_val)

                            save_data['fq3'].append(fq3_val)
                            save_data['time3'].append(t3)   
                            del rho3
                            monitor_memory()

                        if methods[3]:                        
                            rho4, t4 = kdq.QST_LRE()
                            #print('t4:',t4)

                            start_time = time.time()       
                            fq4_val = cal_Fidelity(rho4, kdq.rho).item()
                            if kdq.N >= N_moniter:
                                t = time.time() - start_time #t_stamp
                                print('t_fid4:', t, ',fq4:', fq4_val)

                            save_data['fq4'].append(fq4_val)
                            save_data['time4'].append(t4)   

                            del rho4
                            monitor_memory()

                        del kdq     
                        monitor_memory()                                                                              
                        break
                    except:
                        #print(f'-----------Error Error!!! rank={args.rank}----------')
                        traceback.print_exc()
                        raise RuntimeError(f'Error encountered with rank={args.rank}')
                        #continue           
                    
                    #print(prof.key_averages().table(sort_by="cuda_memory_usage"))      # print GPU memory usage profile      

            
            print(f'noise={args.noise_strength}, N={args.N}\n ')
            for key, value in save_data.items():
                print(f'{key}: {value}')
            save_file(save_data, [args.noise_strength, args.state_flag], [args.N])


# fidelity with the number of shots of different qubits
def exp_2(args):
    args.state_flag = 'random_Rank'    # {'random_Rank', 'Werner', 'W', 'GHZ'}
    args.exp_flag = 2
    args.mea_flag = 'kd'

    methods = [1, 1, 1, 1]
    exp = 10
    for args.N in [1,2,3,4,5]:  
        if args.N in [1,2,3]:
            shots_values = [10**i for i in range(2, 7)]
        elif args.N in [3, 4, 5, 6, 7]:
            shots_values = [10**i for i in range(2, 9)]
        for args.shots in shots_values:  
            save_data = {
            'fq1': [],
            'fq2': [],
            'fq3': [],         
            'fq4': [],
            }
            for _ in range(exp):
                if args.state_flag == 'random_Rank':
                    args.rank = np.random.randint(1, 2**args.N + 1)
                elif args.state_flag == 'Werner':
                    purity_flag = 1
                    while purity_flag:
                        args.purity = np.random.random()
                        if args.purity >= 1/2**args.N:
                            purity_flag = 0
                elif args.state_flag == 'W':
                    args.purity = 1
                
                # initialize KDQ object and perform measurement
                kdq = KDQ(args)                    
                kdq.cir_pef()

                # Execute quantum state tomography methods, save fidelity and time
                method_funcs = [kdq.QST_LIE, kdq.QST_MLE, kdq.QST_CGAPG, kdq.QST_LRE]
                for idx, flag in enumerate(methods):
                    if flag:
                        rho, _ = method_funcs[idx]()
                        save_data[f'fq{idx+1}'].append(cal_Fidelity(rho, kdq.rho).item())
                        #save_data[f'time{idx+1}'].append(t)                                                                                        
                
                
            print(f'N={args.N}, shots={args.shots}, rank={args.rank}', save_data,'\n')     
            save_file(save_data, [args.shots, args.state_flag], [args.N])   



# the fidelity with the strength of gate noise/ readout error
def exp_3(args):
    args.exp_flag = 1
    args.noise_strength = 0
    args.mea_flag = 'kd'
    args.state_flag = 'random_Rank'
    args.purity = 1 
    args.rank = 1
    args.error_flag = ['depolarizing', 'readouterror', 'bitflip', 'phaseflip']  # 'depolarizing', 'readouterror', 'bitflip', 'phaseflip'

    qubits_range = range(1,3)  
    exp = 10  # Number of repetitions for each noise strength
    methods = [1,1]

    # Loop over different numbers of qubits.
    for args.N in qubits_range:
        # Set shots closest to 10**(N+3)
        target = 10 ** (args.N + 3)
        m = round(math.log2(target))
        #m = 8 + 3 * args.N
        args.shots = 2 ** m
        #print(f'shots for {args.N}', args.shots)

        # You may try different decompositions if needed.
        for args.decom_flag in [0]:   # [0,1]
            noise_strengths = np.logspace(-4, 0, 5)  #np.logspace(-5, 0, 6)

            # Loop over various noise strengths.
            for noise in noise_strengths:
                args.noise_strength = noise

                # Lists to collect fidelity for each method.
                lie_list = []
                mle_list = []

                for _ in range(exp):
                    while True:
                        try:                    
                            kdq = KDQ(args)
                            #print('rho=', kdq.rho)
                            kdq.cir_pef()

                            # QST_LIE
                            if methods[0]:
                                rho_lie, _ = kdq.QST_LIE()
                                fq_lie = cal_Fidelity(rho_lie, kdq.rho).item()
                                lie_list.append(fq_lie)
                                del rho_lie

                            # QST_MLE
                            if methods[1]:
                                rho_mle, _ = kdq.QST_MLE()
                                fq_mle = cal_Fidelity(rho_mle, kdq.rho).item()
                                mle_list.append(fq_mle)
                                del rho_mle

                            del kdq
                            break
                        except:
                            traceback.print_exc()
                            raise RuntimeError(f'Error encountered')
                            #print(f'Error encountered')

                print(f'N={args.N}, noise_strength={noise:.5f}: '
                      f'LIE: {lie_list}, '
                      f'MLE: {mle_list}')


# loss function and fidelity with the number of iteration
def exp_4(args):
    args.state_flag = 'random_Rank' #'random_Rank', 'Werner', 'W'
    args.mea_flag = 'kd'
    args.n_epochs = 50

    args.noise_strength = 0
    methods = [1, 1] 
    exp = 10

    qubit_range = [6] #[6,8,10,12]
    args.his_flag = 1

    save_data = {
    'fq3': [],
    'fq4': [],           
    }

    for args.N in qubit_range:
        for _ in range(exp):
            if args.state_flag == 'random_Rank':
                args.rank = torch.randint(1, 2**args.N + 1, (1,), device=args.device)[0].item()
            elif args.state_flag == 'Werner':
                purity_flag = 1
                while purity_flag:
                    args.purity = torch.rand(1, device=args.device).item()
                    if args.purity >= 1/2**args.N:
                        purity_flag = 0
            elif args.state_flag == 'W':
                args.purity = 1
            
            while True:
                try:
                    kdq = KDQ(args)                                      
                    kdq.cir_pef()

                    if methods[0]:      
                        fq_his3 = kdq.QST_MLE()
                        save_data['fq3'].append(fq_his3)         

                    if methods[1]:                        
                        fq_his4 = kdq.QST_CGAPG()
                        save_data['fq4'].append(fq_his4[:args.n_epochs+1])     

                    #for key, value in save_data.items():
                    #    print(f'{key}: {value}')
                    del kdq                                     
                    monitor_memory()                                              
                    break
                except:
                    traceback.print_exc()
                    raise RuntimeError(f'Error encountered with rank={args.rank}')

        # Plot fidelity values
        fq_data = {}
        if len(save_data['fq3']) > 0:
            fq3_stack = np.stack(save_data['fq3'], axis=0)
            fq_data['fq3'] = fq3_stack
        if len(save_data['fq4']) > 0:
            fq4_stack = np.stack(save_data['fq4'], axis=0)
            fq_data['fq4'] = fq4_stack
        save_file(fq_data, [args.noise_strength, args.state_flag], [args.N, "fidelity_iterations"])



# fidelity with the rank and purity
def exp_5(args):
    args.state_flag = 'random_Rank' #, 'Werner', 'W'
    args.mea_flag = 'kd'
    args.rankflag = 0

    for args.noise_strength in [1e-5]:         #[1e-5, 1e-4, 1e-2]: 
        methods = [1, 1]  
        exp = 10   #if args.N <=12 else 2

        #for log2rank in np.array([np.random.random()]):   #np.array([0, 0.4, 0.7, 1]):
        #    args.rank = int(2**(args.N * log2rank))
        for args.rank in list(range(1, 2**args.N + 1, 50)):   #list(range(1, 2**args.N+1, (2 ** args.N)//10))+[2**args.N]:  
        #for args.purity in 0.1*np.array(range(1, 11)):    #[1, 0.6, 0.3]: / 0.1*np.array(range(1, 11))  0.1*torch.arange(1, 11)
            save_data = {
            'fq1': [],
            'time1': [],
            'fq3': [],
            'time3': [],               
            }
            for _ in range(exp):
                while True:
                    try:
                        kdq = KDQ(args)                           
                        kdq.cir_pef()

                        if methods[0]:
                            rho1, t1 = kdq.QST_LIE()
                            kdq.rho = kdq.rho.to(kdq.device)   

                            fq1_val = cal_Fidelity(rho1, kdq.rho).item()
                            save_data['fq1'].append(fq1_val)

                            save_data['time1'].append(t1)   
                            del rho1 

                        if methods[1]:                        
                            #rho3, t3, cost_his, fq_his = kdq.QST_MLE()
                            rho3, t3 = kdq.QST_MLE()
                            fq3_val = cal_Fidelity(rho3, kdq.rho).item()
                            save_data['fq3'].append(fq3_val)
                            save_data['time3'].append(t3)   
                            #save_data['fq3'].append(fq_his)
                            #save_data['cost3'].append(cost_his)
                            #print('t3:',t3)
                            del rho3
                        
                        del kdq                                                                                   
                        break
                    except:
                        #print(f'-----------Error Error!!! rank={args.rank}----------')
                        traceback.print_exc()
                        raise RuntimeError(f'Error encountered with rank={args.rank}')
                        #continue           
                    
                    # Print memory usage if needed
                    #print(prof.key_averages().table(sort_by="cuda_memory_usage"))        

            
            print(f'\n noise={args.noise_strength}, N={args.N}, rank={args.rank}')
            for key, value in save_data.items():
                print(f'{key}: {value}')
            save_file(save_data, [args.noise_strength, args.state_flag], [args.N, args.rank])  
            #save_file(save_data, [args.noise_strength, args.state_flag], [args.N, args.purity])            



if __name__ == '__main__':
    # ----------parameters----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2, help="number of qubits")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="compute device for torch (auto=use CUDA if available)")
    parser.add_argument("--rank", type=int, default=1, help="rank of the target random mixed state")
    parser.add_argument("--purity", type=float, default=1., help="purity of the target random mixed state")

    parser.add_argument("--state_flag", type=str, default="random_Rank", choices=["GHZ", "Product", "W","Werner", "random_Purity", "random_Rank"], help="name of state in library")
    parser.add_argument("--mea_flag", type=str, default="kd", choices=["kd", "Pauli_povm", "Pauli_normal", "Tetra4"], help="name of measurement basis")    
    parser.add_argument("--exp_flag", type=int, default=0, help="0: get quasiprobabilities from calculation, 1: simulation")    
    parser.add_argument("--parallel_flag", type=int, default=0, choices=[0,1], help="0:prepare and measure scheme, 1:parallel scheme")
    parser.add_argument("--error_flag", type=str, default='gatenoise', choices=['gatenoise', 'readouterror'], help="type of noise in Qiskit simulation")
    parser.add_argument("--decom_flag", type=int, default=1, help="0: donot decompose 3-qubit cswap gates, 1: decompose")
    parser.add_argument("--rankflag", type=int, default=0, help="0: full rank initialization, 1: known rank initialization")
    parser.add_argument("--his_flag", type=int, default=0, help="0: donot record loss function and fidelity with iteration, 1: record")  
    parser.add_argument("--noise_strength", type=float, default=1e-4, help="Gaussian noise strength σd")

    parser.add_argument("--sample_size", type=int, default=int(1e4), help="number of samples used for maximum likelihood reconstruction method")
    parser.add_argument("--shots", type=int, default=8192, help="number of shots used for qiskit simulation")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")

#parser.add_argument("--mea1_eig", type=torch.tensor, default=1000, help="number of epochs of training")


    args = parser.parse_args()
    # Resolve device preference
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested via --device=cuda but no GPU is available.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.device = device

    exp_1(args)

