import numpy as np
import pennylane as qml
from scipy.linalg import expm
import math
import matplotlib.pyplot as plt
plt.style.use('pennylane.drawer.plot')


# =====================================================
# 1. TFIM Hamiltonian
# =====================================================
def kron_n(op_list):
    r = np.array([[1]])
    for op in op_list:
        r = np.kron(r, op)
    return r

def tfim_model(n):
    Z = np.array([[1,0],[0,-1]]); X = np.array([[0,1],[1,0]]); I = np.eye(2)
    H = np.zeros((2**n, 2**n), dtype=np.complex128)
    for i in range(n):
        Zi = [I]*n; Zi[i]=Z; Zi[(i+1)%n]=Z; H -= kron_n(Zi)
        Xi = [I]*n; Xi[i]=X; H -= kron_n(Xi)
    return H

def shift_and_normalize_H(H):
    vals = np.linalg.eigvalsh(H)
    lmin, lmax = vals.min(), vals.max()
    Hn = (H - lmin*np.eye(len(H))) / (lmax - lmin) * (np.pi/2)
    l0 = np.min(np.linalg.eigvalsh(Hn))
    return Hn - l0*np.eye(len(Hn))

def get_ground_state(H):
    vals, vecs = np.linalg.eigh(H)
    return vecs[:, np.argmin(vals)]


# =====================================================
# 2. LCU
# =====================================================
def build_lcu_data(H_shifted, n, m, eps=1e-4):
    dim = 2**n
    k_full = np.arange(-m, m+1)
    alpha_full = 2**(-2*m) * np.array([math.comb(2*m, m+k) for k in k_full])
    keep = np.where(alpha_full > eps)[0]
    k_vals = k_full[keep]
    alpha = alpha_full[keep]
    alpha = alpha / alpha.sum()
    pad_len = 2**int(np.ceil(np.log2(len(alpha))))
    alpha_padded = np.zeros(pad_len)
    alpha_padded[:len(alpha)] = alpha
    weights = np.sqrt(alpha_padded / alpha_padded.sum())
    n_anc = int(np.log2(pad_len))
    U_list = [expm(2j*k*H_shifted) for k in k_vals] + [np.eye(dim)]*(pad_len-len(k_vals))
    return weights, U_list, n_anc


# =====================================================
# 3.  U = LCUPrepare，O = Oracle
# =====================================================
@qml.prod
def LCUPrepare_op(weights, U_ops, anc_wires, sys_wires):
    """A = StatePrep → Select(U) → StatePrep†"""
    for i in sys_wires:
        qml.Hadamard(i)
    qml.StatePrep(weights, wires=anc_wires)
    qml.Select(
        [qml.QubitUnitary(U, wires=sys_wires) for U in U_ops],
        control=anc_wires
    )
    qml.adjoint(qml.StatePrep)(weights, wires=anc_wires)

@qml.prod
def Oracle_op(weights, U_ops, anc_wires, sys_wires):
    qml.FlipSign(n=0, wires=anc_wires)


# =====================================================
# 4. Amplitude Amplification
# =====================================================
def run_AA_LCU(n=4, m=6, R=1):

    H = tfim_model(n)
    Hs = shift_and_normalize_H(H)
    gs = get_ground_state(Hs)

    weights, U_ops, n_anc = build_lcu_data(Hs, n, m)
    anc = list(range(n_anc))
    sys = list(range(n_anc, n_anc+n))
    all_wires = anc + sys
    dev = qml.device("default.qubit")
    @qml.qnode(dev)
    def circuit(iters):

        LCUPrepare_op(weights, U_ops, anc, sys)
        qml.AmplitudeAmplification(
            U = LCUPrepare_op(weights, U_ops, anc, sys),
            #U = prepare_init(n),
            O = Oracle_op(weights, U_ops, anc, sys),
            iters = iters,
            fixed_point=True,
            work_wire=len(all_wires),
        )
        return qml.probs(wires=all_wires)
    #qml.draw_mpl(circuit)(R)[0].show()

    prob = circuit(R)
    return prob



# =====================================================
# 5. test
# =====================================================
# if __name__ == "__main__":
#     fig, axs = plt.subplots(2, 2, figsize=(14, 10))
#     for i in range(0,10,2):
#         output = run_AA_LCU(n=4, m=6, R=i)
#         ax = axs[i // 4, i // 2 % 2]
#         ax.bar(range(len(output)), output)
#         ax.set_ylim(0, 0.6)
#         ax.set_title(f"Iteration {i}")
#
#     plt.tight_layout()
#     plt.subplots_adjust(bottom=0.1)
#     plt.axhline(0, color='black', linewidth=1)
#     plt.show()

if __name__ == "__main__":
    R_list = [0, 1, 3, 5, 7, 9]       # 指定要跑的迭代次数
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.flatten()               # 把 2×3 的 axes 摊平方便索引

    for i, R in enumerate(R_list):
        output = run_AA_LCU(n=4, m=6, R=R)
        ax = axs[i]
        ax.bar(range(len(output)), output)
        ax.set_ylim(0, 0.3)
        ax.set_title(f"Iteration R = {R}")
        ax.axhline(0, color='black', linewidth=1)

    plt.tight_layout()
    plt.show()


