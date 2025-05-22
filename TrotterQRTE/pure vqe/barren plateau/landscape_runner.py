
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ===== Ansatz definitions =====
def ansatz_hva(params, nqubits, depth):
    qc = QuantumCircuit(nqubits)
    t = 0
    for i in range(nqubits):
        qc.h(i)
    for _ in range(depth):
        for i in range(nqubits):
            j = (i + 1) % nqubits
            qc.rzz(params[t], i, j)
            t += 1
        for i in range(nqubits):
            qc.rx(params[t], i)
            t += 1
    return qc

def ansatz_hea(params, nqubits, depth):
    qc = QuantumCircuit(nqubits)
    t = 0
    for _ in range(depth):
        for i in range(nqubits):
            qc.ry(params[t], i)
            t += 1
        for i in range(nqubits - 1):
            qc.cx(i, i + 1)
    return qc

# ===== Hamiltonians =====
def get_global_H(nqubits):
    return SparsePauliOp("Z" * nqubits, coeffs=[1.0])

def get_local_H(nqubits):
    terms = []
    for i in range(nqubits - 1):
        label = ['I'] * nqubits
        label[i], label[i + 1] = 'Z', 'Z'
        terms.append(("".join(label), -1.0))
    for i in range(nqubits):
        label = ['I'] * nqubits
        label[i] = 'X'
        terms.append(("".join(label), -1.0))
    return SparsePauliOp.from_list(terms)

# ===== Noise and Simulator =====
def build_noise_model(error_rate):
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate, 1), ['rx', 'ry', 'id'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate * 10, 2), ['cx', 'rzz'])
    return noise_model

def get_simulator(error_rate):
    return AerSimulator(method='density_matrix', noise_model=build_noise_model(error_rate))

# ===== Landscape generation =====
def generate_surface_from_ansatz(nqubits, depth, H, error_rate, ansatz_type, x_idx, y_idx, step=0.25):
    X, Y = np.meshgrid(np.arange(-np.pi, np.pi, step), np.arange(-np.pi, np.pi, step))
    Z = np.zeros_like(X)

    if ansatz_type == "hea":
        total_params = nqubits * depth
    else:
        total_params = 2 * nqubits * depth

    base_params = np.random.uniform(-np.pi, np.pi, total_params)

    noisy = error_rate > 0
    if noisy:
        simulator = get_simulator(error_rate)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            thetas = base_params.copy()
            thetas[x_idx] = X[i, j]
            thetas[y_idx] = Y[i, j]

            if ansatz_type == "hea":
                qc = ansatz_hea(thetas, nqubits, depth)
            else:
                qc = ansatz_hva(thetas, nqubits, depth)

            if noisy:
                qc.save_density_matrix()
                tqc = transpile(qc, simulator)
                result = simulator.run(tqc).result()
                rho = result.data(0)['density_matrix']
                val = np.real(rho.expectation_value(H))
            else:
                state = Statevector.from_instruction(qc)
                val = np.real(state.expectation_value(H))
            Z[i, j] = val

    return X, Y, Z

# ===== Plotting =====
def plot_and_save_surface(X, Y, Z, title, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=False)
    ax.set_title(title)
    ax.set_xlabel("θ_x")
    ax.set_ylabel("θ_y")
    ax.set_zlabel("Cost")
    ax.set_zlim(-4, 4)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# ===== Main Experiment Entry Point =====
if __name__ == "__main__":
    nqubits = 4
    depth = nqubits - 1
    error_rate = 0.001
    ansatz_type = "hea"  # or "hea"
    H = get_local_H(nqubits)#get_global_H(nqubits)#get_local_H(nqubits)  # or get_global_H(nqubits)

    if ansatz_type == "hea":
        total_params = nqubits * depth
        x_idx, y_idx = total_params - 2, total_params - 1
    else:
        total_params = 2 * nqubits * depth
        x_idx, y_idx = total_params - 2, total_params - 1

    X, Y, Z = generate_surface_from_ansatz(nqubits, depth, H, error_rate, ansatz_type, x_idx, y_idx)

    title = f"{ansatz_type.upper()} Cost Landscape (n={nqubits}, error={error_rate})"
    filename = f"landscape_{ansatz_type}_n{nqubits}_error{error_rate}.png"
    plot_and_save_surface(X, Y, Z, title, filename)
