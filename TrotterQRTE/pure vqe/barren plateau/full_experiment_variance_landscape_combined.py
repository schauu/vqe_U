
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ==== Hamiltonians ====
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

# ==== Ansatz ====
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
    idx = 0
    for d in range(depth):
        for i in range(nqubits):
            qc.ry(params[idx], i)
            idx += 1
        for i in range(nqubits - 1):
            qc.cx(i, i + 1)
    return qc

# ==== Noise ====
def build_noise_model(error_rate):
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate, 1), ['rx', 'ry', 'id'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate * 10, 2), ['cx', 'rzz'])
    return noise_model

def get_simulator(error_rate):
    return AerSimulator(method='density_matrix', noise_model=build_noise_model(error_rate))

# ==== Gradient + Expectation ====
def expectation_density_matrix(thetas, nqubits, depth, H, simulator, ansatz_type):
    if ansatz_type == "hea":
        qc = ansatz_hea(thetas, nqubits, depth)
    else:
        qc = ansatz_hva(thetas, nqubits, depth)
    qc.save_density_matrix()
    tqc = transpile(qc, simulator)
    result = simulator.run(tqc).result()
    rho = result.data(0)['density_matrix']
    return np.real(rho.expectation_value(H))

def compute_gradient(thetas, nqubits, depth, H, simulator, target_idx, ansatz_type, eps=1e-4):
    shifted_plus = thetas.copy()
    shifted_minus = thetas.copy()
    shifted_plus[target_idx] += eps
    shifted_minus[target_idx] -= eps
    f_plus = expectation_density_matrix(shifted_plus, nqubits, depth, H, simulator, ansatz_type)
    f_minus = expectation_density_matrix(shifted_minus, nqubits, depth, H, simulator, ansatz_type)
    return (f_plus - f_minus) / (2 * eps)

# ==== Landscape ====
def ansatz_surface(rotations, nqubits):
    qc = QuantumCircuit(nqubits)
    for i in range(nqubits):
        qc.rx(rotations[0][i], i)
        qc.ry(rotations[1][i], i)
    return qc

def generate_surface(nqubits, H, error_rate):
    Z = []
    X = np.arange(-np.pi, np.pi, 0.25)
    Y = np.arange(-np.pi, np.pi, 0.25)
    X, Y = np.meshgrid(X, Y)

    noisy = error_rate > 0
    if noisy:
        simulator = get_simulator(error_rate)

    for x in X[0, :]:
        Z_row = []
        for y in Y[:, 0]:
            rotations = [[x] * nqubits, [y] * nqubits]
            qc = ansatz_surface(rotations, nqubits)
            if noisy:
                qc.save_density_matrix()
                tqc = transpile(qc, simulator)
                result = simulator.run(tqc).result()
                rho = result.data(0)['density_matrix']
                val = np.real(rho.expectation_value(H))
            else:
                state = Statevector.from_instruction(qc)
                val = np.real(state.expectation_value(H))
            Z_row.append(val)
        Z.append(Z_row)

    return X, Y, np.array(Z)

def plot_and_save_surface(X, Y, Z, title, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=False)
    ax.set_title(title)
    ax.set_zlim(-4, 4)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# ==== Main ====
if __name__ == "__main__":
    #ansatz_type = "hea"  # or "hva"
    ansatz_type = 'hva'
    nqubits_list = [4, 6, 8, 10]
    error_rate = 0.1
    trials = 100
    variances_global = []
    variances_local = []

    # for ham_type in ["global", "local"]:
    #     variances = []
    #     for nqubits in nqubits_list:
    #         depth = nqubits - 1
    #         if ansatz_type == "hea":
    #             total_params = nqubits * depth
    #             target_idx = nqubits * (depth - 1) + 1
    #         else:
    #             total_params = 2 * nqubits * depth
    #             target_idx = 2 * nqubits * (depth - 1) + 1
    #
    #         H = get_global_H(nqubits) if ham_type == "global" else get_local_H(nqubits)
    #         simulator = get_simulator(error_rate)
    #         grads = []
    #
    #         for _ in range(trials):
    #             thetas = np.random.uniform(-np.pi, np.pi, total_params)
    #             grad = compute_gradient(thetas, nqubits, depth, H, simulator, target_idx, ansatz_type)
    #             grads.append(np.abs(grad))
    #
    #         variance = np.var(grads)
    #         variances.append(variance)
    #         print(f"[{ham_type.upper()} | {ansatz_type.upper()}] n={nqubits}: Var={variance:.4e}")
    #
    #     if ham_type == "global":
    #         variances_global = variances
    #     else:
    #         variances_local = variances
    #
    # #Plot gradient variance
    # plt.figure(figsize=(8, 5))
    # plt.plot(nqubits_list, variances_global, 'o-', label="Global Cost")
    # plt.plot(nqubits_list, variances_local, 's--', label="Local Cost")
    # plt.yscale("log")
    # plt.xlabel("Number of Qubits")
    # plt.ylabel("Gradient Variance (log scale)")
    # plt.title(f"Gradient Variance vs Qubits ({ansatz_type.upper()}, noise={error_rate})")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"gradient_variance_{ansatz_type}.png")
    # plt.show()

    # Cost landscapes (n = 4)
    n_land = 4
    H_land = get_global_H(n_land)
    # X1, Y1, Z1 = generate_surface(n_land, H_land, error_rate=0.0)
    # plot_and_save_surface(X1, Y1, Z1, f"Ideal Global Landscape (n={n_land})", f"landscape_global_ideal_n{n_land}.png")

    X2, Y2, Z2 = generate_surface(n_land, H_land, error_rate=error_rate)
    plot_and_save_surface(X2, Y2, Z2, f"Noisy Global Landscape (n={n_land}, error={error_rate})", f"landscape_global_noisy_n{n_land}.png")

    # === Local H landscape ===
    H_land_local = get_local_H(n_land)

    # X3, Y3, Z3 = generate_surface(n_land, H_land_local, error_rate=0.0)
    # plot_and_save_surface(X3, Y3, Z3, f"Ideal Local Landscape (n={n_land})", f"landscape_local_ideal_n{n_land}.png")

    X4, Y4, Z4 = generate_surface(n_land, H_land_local, error_rate=error_rate)
    plot_and_save_surface(X4, Y4, Z4, f"Noisy Local Landscape (n={n_land}, error={error_rate})", f"landscape_local_noisy_n{n_land}.png")
