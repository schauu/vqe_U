#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

def build_tfim_hamiltonian_4x3():
    nrow, ncol = 4, 3
    nqubits = nrow * ncol
    pauli_terms = []
    coeffs = []

    def idx(r, c):
        return r * ncol + c

    for r in range(nrow):
        for c in range(ncol):
            q = idx(r, c)
            if c < ncol - 1:
                q_right = idx(r, c + 1)
                zz = ['I'] * nqubits
                zz[q], zz[q_right] = 'Z', 'Z'
                pauli_terms.append(''.join(reversed(zz)))
                coeffs.append(1.0)
            if r < nrow - 1:
                q_down = idx(r + 1, c)
                zz = ['I'] * nqubits
                zz[q], zz[q_down] = 'Z', 'Z'
                pauli_terms.append(''.join(reversed(zz)))
                coeffs.append(1.0)

    for i in range(nqubits):
        x = ['I'] * nqubits
        x[i] = 'X'
        pauli_terms.append(''.join(reversed(x)))
        coeffs.append(-1.0)

    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))

def ansatz_hamiltonian(nqubits, depth, params):
    circuit = QuantumCircuit(nqubits)
    t = 0
    for i in range(nqubits):
        circuit.h(i)
    for _ in range(depth):
        for i in range(nqubits):
            j = (i + 1) % nqubits
            circuit.rzz(params[t], i, j)
            t += 1
        circuit.barrier()
        for i in range(nqubits):
            circuit.rx(params[t], i)
            t += 1
    return circuit

def cost_function2(params, nqubits, depth, error_rate):
    hamiltonian = build_tfim_hamiltonian_4x3()
    circuit = ansatz_hamiltonian(nqubits, depth, params)
    circuit = circuit.decompose()
    simulator = AerSimulator()
    circuit.save_statevector()
    result = simulator.run(circuit).result()
    u = result.data(0)['statevector'].data
    expectation = (u.conj().dot(hamiltonian.to_matrix())).dot(u)
    return expectation.real

def make_callback(tracker, nqubits, depth, error_rate):
    def callback(xk):
        energy = cost_function2(xk, nqubits, depth, error_rate)
        tracker.append(energy)
    return callback

def run_vqe(nqubits, depth, error_rate):
    np.random.seed(42)
    nparams = 2 * nqubits * depth
    params0 = np.random.random(nparams)
    tracker = []
    callback_fn = make_callback(tracker, nqubits, depth, error_rate)
    result = minimize(cost_function2, params0, args=(nqubits, depth, error_rate),
                      method="BFGS", tol=1e-5, callback=callback_fn)
    return result.fun, tracker

def main():
    depths = [1, 2, 3, 4]
    nqubits = 12
    error_rate = 0
    results = {}

    for d in depths:
        final_energy, trajectory = run_vqe(nqubits, d, error_rate)
        results[d] = trajectory
        print(f"depth={d}, final energy={final_energy:.6f}, iterations={len(trajectory)}")

    plt.figure(figsize=(10, 6))
    for d, traj in results.items():
        plt.plot(traj, label=f"depth={d}")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("VQE Optimization Trajectories for Different Depths (HEA, TFIM 4x3)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("vqe_tfim4x3_depths.png")

if __name__ == "__main__":
    main()
