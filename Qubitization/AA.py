import numpy as np
import pennylane as qml
from scipy.linalg import expm
import math
import matplotlib.pyplot as plt


# =====================================================
# 1. TFIM Hamiltonian + 辅助函数
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
# 2. LCU 构造
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
# 3. 定义 U = LCUPrepare，O = Oracle
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

# @qml.prod
# def prepare_init(n):
#     for i in range(n):
#         qml.Hadamard(wires=i)

@qml.prod
def Oracle_op(weights, U_ops, anc_wires, sys_wires):
    """蓝色块：A → FlipSign(anc=0) → A†"""
    #LCUPrepare_op(weights, U_ops, anc_wires, sys_wires)
    qml.FlipSign(n=0, wires=anc_wires)
    #qml.adjoint(LCUPrepare_op)(weights, U_ops, anc_wires, sys_wires)


# =====================================================
# 4. 主过程：Amplitude Amplification
# =====================================================
def run_AA_LCU(n=4, m=6, R=1):
    """运行一次 amplitude amplification for LCU"""
    H = tfim_model(n)
    Hs = shift_and_normalize_H(H)
    gs = get_ground_state(Hs)

    weights, U_ops, n_anc = build_lcu_data(Hs, n, m)
    #print('Weights:', weights)
    #print('U_ops:', U_ops)
    anc = list(range(n_anc))
    sys = list(range(n_anc, n_anc+n))
    all_wires = anc + sys
    dev = qml.device("default.qubit", wires=len(all_wires)+1, shots=None)
    #dev = qml.device("default.qubit")
    @qml.qnode(dev)
    def circuit(iters):
        # for i in range(n):
        #     qml.Hadamard(wires=i)
        LCUPrepare_op(weights, U_ops, anc, sys)
        qml.AmplitudeAmplification(
            U = LCUPrepare_op(weights, U_ops, anc, sys),
            #U = prepare_init(n),
            O = Oracle_op(weights, U_ops, anc, sys),
            iters = iters,
            fixed_point=True,
            work_wire=len(all_wires)+1,
        )
        return qml.state()
    #qml.draw_mpl(circuit)(R)[0].show()
    psi = circuit(R)
    #psi = psi.reshape((2**n_anc, 2**n))
    psi = psi.reshape((2 ** (n_anc+1), 2 ** n))
    sys_amp = psi[0, :]
    #print('System state', sys_amp)
    success_prob = np.real(np.vdot(sys_amp, sys_amp))
    #success_prob = np.linalg.norm(sys_amp)
    normed = sys_amp / np.linalg.norm(sys_amp) if np.linalg.norm(sys_amp) > 0 else sys_amp
    fid = np.abs(np.vdot(normed, gs))**2
    return fid, success_prob



if __name__ == "__main__":
    R_list = list(range(11))
    fidelities = []
    success_probs = []

    print("R | Fidelity | Success Prob")
    print("-"*32)

    for R in R_list:
        fid, p = run_AA_LCU(n=4, m=6, R=R)
        fidelities.append(fid)
        success_probs.append(p)
        print(f"{R:>2d} | {fid:8.4f} | {p:10.4f}")

    # === 绘图 ===
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(R_list, fidelities, marker='o', label="Fidelity", color='tab:orange')
    ax1.set_xlabel("Iteration R")
    ax1.set_ylabel("Fidelity", color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:orange')
    ax1.set_ylim(0, 1.05)

    # 第二个坐标轴绘制成功概率
    ax2 = ax1.twinx()
    ax2.plot(R_list, success_probs, marker='s', label="Success Probability", color='tab:blue')
    ax2.set_ylabel("Success Probability", color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0, 1.05)

    # 图例
    fig.legend(loc='lower right', bbox_to_anchor=(0.9, 0.05))
    fig.tight_layout()
    #plt.title("Amplitude Amplification Performance vs Iteration")
    plt.show()

