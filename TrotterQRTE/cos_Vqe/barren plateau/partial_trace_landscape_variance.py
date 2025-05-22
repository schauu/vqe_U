
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def ansatz_1(nqubits, depth, params):
    circuit = QuantumCircuit(nqubits + 1)
    t = 0
    for _ in range(depth):
        for i in range(nqubits + 1):
            circuit.ry(params[t], i)
            t += 1
        circuit.barrier()
        for i in range(nqubits):
            circuit.cry(params[t], i, i + 1)
            t += 1
    return circuit

def run_sim(params, nqubits, depth):
    qc = ansatz_1(nqubits, depth, params)
    return Statevector.from_instruction(qc).data

def cost_func(params, nqubits, depth):
    statevector = run_sim(params, nqubits, depth)
    dim = 2 ** nqubits
    prob = np.linalg.norm(statevector[:dim]) ** 2
    return 1 - prob

def generate_surface(nqubits, depth, x_idx, y_idx, step=0.5):
    total_params = depth * (2 * nqubits + 1)
    X, Y = np.meshgrid(np.arange(-np.pi, np.pi, step), np.arange(-np.pi, np.pi, step))
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            thetas = np.random.uniform(-np.pi, np.pi, size=total_params)
            thetas[x_idx] = X[i, j]
            thetas[y_idx] = Y[i, j]
            Z[i, j] = cost_func(thetas, nqubits, depth)
    return X, Y, Z

def compute_gradient(params, idx, eps, nqubits, depth):
    plus = params.copy()
    minus = params.copy()
    plus[idx] += eps
    minus[idx] -= eps
    return (cost_func(plus, nqubits, depth) - cost_func(minus, nqubits, depth)) / (2 * eps)

def run_variance(nqubits_list, depth, target_idx, trials=100):
    variances = []
    for n in nqubits_list:
        total_params = depth * (2 * n + 1)
        grads = []
        for _ in range(trials):
            params = np.random.uniform(-np.pi, np.pi, size=total_params)
            grad = compute_gradient(params, target_idx, 1e-4, n, depth)
            grads.append(grad)
        variances.append(np.var(grads))
    return variances

if __name__ == "__main__":
    nqubits = 4
    depth = 2
    x_idx = 3
    y_idx = 7

    X, Y, Z = generate_surface(nqubits, depth, x_idx, y_idx)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title("Cost Landscape (Partial Trace Cost)")
    plt.tight_layout()
    plt.savefig("partial_trace_cost_landscape.png")
    plt.show()

    qubit_list = [2, 4, 6, 8]
    target_param_idx = 5
    var_list = run_variance(qubit_list, depth=2, target_idx=target_param_idx)
    plt.figure()
    plt.plot(qubit_list, var_list, 'o-')
    plt.yscale("log")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Gradient Variance (log scale)")
    plt.title("Gradient Variance (Partial Trace Cost)")
    plt.tight_layout()
    plt.savefig("partial_trace_cost_variance.png")
    plt.show()
