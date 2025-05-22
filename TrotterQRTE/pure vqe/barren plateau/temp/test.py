from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import matplotlib.pyplot as plt

def build_noise_model(error_rate):
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate, 1), ['h', 'id'])
    return noise_model

# 创建两个模拟器（一个有噪声，一个没有）
sim_ideal = AerSimulator()
sim_noisy = AerSimulator(method='density_matrix',noise_model=build_noise_model(0.1))  # 10% noise

# 构建量子电路：只加一个 Hadamard，然后测量
qc = QuantumCircuit(4)
for i in range(4):
    qc.h(i)
qc.measure_all()

# 编译电路
tqc_ideal = transpile(qc, sim_ideal)
tqc_noisy = transpile(qc, sim_noisy)

# 运行
result_ideal = sim_ideal.run(tqc_ideal, shots=100000).result()
result_noisy = sim_noisy.run(tqc_noisy, shots=100000).result()

counts_ideal = result_ideal.get_counts()
counts_noisy = result_noisy.get_counts()

# 输出结果
print("Ideal simulator:", counts_ideal)
print("With noise model:", counts_noisy)

# 可视化
plt.subplot(1, 2, 1)
plt.bar(counts_ideal.keys(), counts_ideal.values(), color='royalblue')
plt.title("Ideal (H gate)")

plt.subplot(1, 2, 2)
plt.bar(counts_noisy.keys(), counts_noisy.values(), color='orangered')
plt.title("Noisy (H gate with depolarizing)")

plt.tight_layout()
plt.show()
