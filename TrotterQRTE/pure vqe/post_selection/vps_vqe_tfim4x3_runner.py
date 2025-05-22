#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

def build_tfim_hamiltonian_4x3():
    nrow, ncol = 4, 3
    nqubits = nrow * ncol
    pauli_terms, coeffs = [], []

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

    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs))), nqubits

def vps_ansatz_rzz_system_only(nqubits, depth, params):
    circuit = QuantumCircuit(nqubits + 1)
    anc = nqubits
    t = 0

    for i in range(nqubits):
        circuit.h(i)

    for _ in range(depth):
        for i in range(nqubits):
            j = (i + 1) % nqubits
            circuit.rzz(params[t], i, j)
            t += 1
        for i in range(nqubits):
            circuit.rx(params[t], i)
            t += 1
        circuit.barrier()

    circuit.ry(params[t], anc)
    t += 1
    circuit.rz(params[t], anc)
    t += 1
    circuit.ry(params[t], anc)

    return circuit

def cost_vps_rzz(params_all, nqubits, depth, hamiltonian):
    n_theta = nqubits * depth * 2
    theta_all = params_all[:n_theta]
    phi_anc = params_all[-3:]

    circuit = vps_ansatz_rzz_system_only(nqubits, depth, list(theta_all) + list(phi_anc))
    circuit.save_statevector()
    sim = AerSimulator()
    result = sim.run(circuit).result()
    u = result.data(0)['statevector'].data

    system_dim = 2 ** nqubits
    psi_sys = np.zeros(system_dim, dtype=complex)
    norm = 0
    for i in range(len(u)):
        b = format(i, f'0{nqubits+1}b')
        if b[-1] == '1':
            idx_sys = int(b[:-1], 2)
            psi_sys[idx_sys] += u[i]
            norm += abs(u[i])**2

    if norm < 1e-8:
        return 1e6
    psi_sys /= np.sqrt(norm)
    return np.real(psi_sys.conj().T @ hamiltonian.to_matrix() @ psi_sys)

def run_vqe_vps_rzz(nqubits, depth, hamiltonian):
    np.random.seed(42)
    nparams = nqubits * depth * 2 + 3
    init_params = np.random.random(nparams)
    energy_track = []

    def callback(xk):
        e = cost_vps_rzz(xk, nqubits, depth, hamiltonian)
        energy_track.append(e)

    result = minimize(
        cost_vps_rzz, init_params, args=(nqubits, depth, hamiltonian),
        method="BFGS", tol=1e-5, callback=callback
    )
    return result.fun, energy_track

def main():
    hamiltonian, nqubits = build_tfim_hamiltonian_4x3()
    depths = [1, 2, 3, 4]
    results = {}

    for d in depths:
        final_E, traj = run_vqe_vps_rzz(nqubits, d, hamiltonian)
        results[d] = {
            "final_energy": final_E,
            "trajectory": traj
        }
        print(f"✅ Depth={d}, Final Energy={final_E:.6f}, Steps={len(traj)}")

    plt.figure(figsize=(10, 6))
    for d in depths:
        plt.plot(results[d]["trajectory"], label=f"depth={d}")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("VPS-VQE on TFIM 4x3")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("vps_vqe_tfim4x3_depths.png")

if __name__ == "__main__":
    main()
