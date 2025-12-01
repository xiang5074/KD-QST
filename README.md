# Kirkwood Dirac Quasiprobability enhanced Quantum State Tomography

Implementation of quantum state tomography methods based on Kirkwood–Dirac quasiprobabilities (KDQ), including KDQ measurement simulation, Pauli POVM baselines, and multiple reconstruction algorithms (KDQ-based LIE, KDQ-based MLE, Pauli-based CG-APG, Pauli-based LRE). The code provides torch-first numerical kernels plus optional Qiskit simulators for measurement generation.

This is the official Pytorch implementation of the paper named "Efficient quantum state tomography with two complementary measurements", under review by *Automatica*, and the paper named "Direct Quantum State Tomography Based on Kirkwood-Dirac Quasiprobability", under review by *American Control Conference 2026*.

## Features
- KDQ quasiprobability generation and sampling for prepare-and-measure or parallel schemes (`KDQ.py`, `circuit.py`).
- Reconstruction pipelines: two-step LIE, least-squares (LSE), maximum likelihood (MLE), conjugate-gradient APG, linear regression estimation (LRE), and CGLS (`others/`).
- Utility layers for POVM evaluation, shuffling, fidelity, and large-scale safe dot products (`Basis_Function.py`).
- State factories for GHZ, W, Werner, random purity/rank states (`Basis_State.py`).
- Experiment runners for simulation benchmarks (`main.py`) and superconducting data post-processing (`superconduct_exp.py` template paths retained but now configurable).

## Repository Structure
- `main.py`: Experiment driver for simulated studies across qubits/noise/shots.
- `KDQ.py`: Core KDQ class handling measurement generation and tomography workflows.
- `circuit.py`: Qiskit circuits for KDQ measurements with configurable noise/decomposition.
- `Basis_Function.py`: POVM evaluation, projection, fidelity, shuffling, and helper math.
- `Basis_State.py`: Canonical and random state generation utilities.
- `others/`
  - `qse_apg.py`: CG-APG maximum-likelihood optimizer.
  - `qse_cgls.py`: Conjugate-gradient line-search solver.
  - `LRE.py`: Linear regression estimation utilities.

## Requirements
This code was tested on a machine with a single NVIDIA A100‑SXM4 (80 GB) for 1–15 qubit state reconstruction, and requires (for comparable large‑scale runs; for smaller problems you can relax the GPU requirements or run on CPU with longer runtimes):
- Python 3.12
- Conda3
- cuda 12.4
- magma==2.9.0
- matplotlib==3.10.6
- qiskit==2.1.2
- qiskit_aer==0.17.1
- pylatexenc==2.10
- pynvml==13.0.1
- numpy==2.3.2
- torch==2.5.0
- scipy==1.16.1


Install the basics:
```bash
pip install cupy-cuda12x matplotlib qiskit_aer pylatexenc pynvml
conda install -c pytorch magma
```

## Quick Start (Simulation)
1. Set CUDA if available: `export CUDA_VISIBLE_DEVICES=...` (optional).
2. Choose an experiment in the main.py file (see below) and run it; results are saved under `results/` in the repo root.
   - `exp_1`: Fidelity/runtime vs. qubit count (KDQ LIE/MLE + Pauli CG-APG/LRE)
     ```bash
     python main.py --exp_flag 0
     ```
   - `exp_2`: Fidelity vs. shots across qubits (KDQ LIE)
     ```bash
     python main.py --exp_flag 2
     ```
   - `exp_3`: Fidelity vs. gate/readout noise strength on Qiskit platform (KDQ LIE, MLE)
     ```bash
     python main.py --exp_flag 1 --error_flag gatenoise
     ```
   - `exp_4`: Infidelity vs. iteration (KDQ MLE, Pauli CG-APG)
     ```bash
     python main.py --exp_flag 0 --his_flag 1 --n_epochs 50
     ```
   - `exp_5`: Fidelity vs. rank for random states (KDQ LIE, KDQ MLE)
     ```bash
     python main.py --exp_flag 0 --N 10
     ```




## Notes
- Default dtype switches to complex128 for moderate qubit counts; large-N paths fall back to complex64 to reduce memory pressure.
- Some experiments may be GPU-memory heavy for N ≥ 14; monitor usage with `monitor_memory()` utilities.

## Citation
If you use this codebase, please cite:
- Efficient quantum state tomography with two complementary measurements.
- Direct Quantum State Tomography Based on Kirkwood-Dirac Quasiprobability.

## License
This code is distributed under an Mozilla Public License Version 2.0.
