import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

def get_ground_state(H):
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs[:, np.argmin(eigvals)]

def shift_and_normalize_H(H):
    eigvals = np.linalg.eigvalsh(H)
    λ_min, λ_max = np.min(eigvals), np.max(eigvals)
    H_norm = (H - λ_min * np.eye(len(H))) / (λ_max - λ_min)
    λ0 = np.min(np.linalg.eigvalsh(H_norm))
    H_shifted = H_norm - λ0 * np.eye(len(H))
    return H_shifted

def cosine_filter_matrix(H, m):
    """Computes cos^{2m}(H) via (cos H)^{2m}"""
    U = expm(1j * H)
    U_dag = expm(-1j * H)
    cosH = 0.5 * (U + U_dag)
    return np.linalg.matrix_power(cosH, 2 * m)

def custom_filter_matrix(H, N):
    """Computes f(H) = (1/N) sum_{j=N}^{2N} e^{-iHj}"""
    return sum(expm(-1j * H * i) for i in range(N, 2 * N + 1)) / N
    #return sum(expm(-1j * H * j) for j in range(-N, N + 1)) #/ N
    #return sum(expm(-1j * H * i) for i in range(N)) / N

def evaluate_fidelity(filter_matrix, init_state, ground_state):
    psi = filter_matrix @ init_state
    psi /= np.linalg.norm(psi)
    return np.abs(np.vdot(psi, ground_state)) ** 2

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

n = 10
dim = 2 ** n
H = heisenberg_xyz_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=1.0)


# 归一化并 shift
H_shifted = shift_and_normalize_H(H)
ground_state = get_ground_state(H_shifted)
init_state = np.zeros(dim)
init_state[int('0101', 2)] = 1.0  # 手动设为 |0101⟩

# Fidelity for custom filter as N varies
max_N = 400
fidelity_list = []
for N_iter in range(1, max_N + 1):
    fH = custom_filter_matrix(H_shifted, N_iter)
    fidelity_custom = evaluate_fidelity(fH, init_state, ground_state)
    fidelity_list.append(fidelity_custom)
# Fidelity for cosine filter
fidelity_cos_list = []
for m_vals in range(1,201):
    cos_filter = cosine_filter_matrix(H_shifted, m_vals)
    fidelity_cos = evaluate_fidelity(cos_filter, init_state, ground_state)
    fidelity_cos_list.append(fidelity_cos)

# 画图
plt.plot(range(1, max_N + 1), fidelity_list, label="Custom Filter")
plt.plot(list(range(2, 401, 2)), fidelity_cos_list, marker='^', label='Direct Matrix')
plt.xlabel("N")
plt.ylabel("Fidelity")
plt.title("Fidelity vs N for XZY model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()