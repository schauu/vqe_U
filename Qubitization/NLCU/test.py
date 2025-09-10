import numpy as np
import math
import pennylane as qml
from scipy.linalg import expm
import matplotlib.pyplot as plt


n =  4
dim = 2 ** n

init_state = np.zeros(dim)
init_state[int('0101', 2)] = 1.0  # 手动设为 |0101⟩

#init_state = np.ones(dim) / np.sqrt(dim)

# init_state = np.zeros(dim)
# init_state[0] = 1

Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
I = np.eye(2)

def heisenberg_xyz_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=1.0):
    dim = 2 ** n
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    def kron_n(op_list):
        result = np.array([[1]])
        for op in op_list:
            result = np.kron(result, op)
        return result

    H = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(n):
        for op, J in zip([X, Y, Z], [Jx, Jy, Jz]):
            ops = [I] * n
            ops[i] = op
            ops[(i + 1) % n] = op  # 周期性边界条件
            H += J * kron_n(ops)

    return H
H = heisenberg_xyz_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=1.0)
eigvals, eigvecs = np.linalg.eigh(H)
λ_min, λ_max = np.min(eigvals), np.max(eigvals)
H_norm = (H - λ_min * np.eye(dim)) / (λ_max - λ_min)
λ0 = np.min(np.linalg.eigvalsh(H_norm))
H_shifted = H_norm - λ0 * np.eye(dim)
ground_state = np.linalg.eigh(H_shifted)[1][:, 0]
H_shifted1 = H_norm/4 - λ0 * np.eye(dim)
ground_state1 = np.linalg.eigh(H_shifted1)[1][:,0]
print(np.allclose(ground_state, ground_state1))