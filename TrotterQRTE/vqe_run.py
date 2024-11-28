#!/usr/bin/env python3
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity, Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.linalg import expm, cosm
import warnings
import sys

# !/usr/bin/env python3
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity, Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.linalg import expm, cosm
import warnings
import sys


# warnings.simplefilter("ignore", np.ComplexWarning)
# np.set_printoptions(linewidth=100)

## Pure VQE
class VQE_Runner:
    def __init__(self, nqubits, error_rate, depth, periodic, parameter):
        self.nqubits = nqubits
        self.error_rate = error_rate
        self.depth = depth
        self.periodic = periodic
        self.parameter = parameter

    def make_heis1(self):
        ops = []
        coeffs = []
        if self.periodic == True:
            for i in range(self.nqubits):
                if i < (self.nqubits - 1):
                    j = i + 1
                else:
                    j = 0
                zs = ["I"] * self.nqubits
                zs[i] = "Z"
                zs[j] = "Z"
                Zop = "".join(zs)
                ops.append(Zop)
                coeffs.append(1 / np.sqrt(2))

                xs = ["I"] * self.nqubits
                xs[i] = "X"
                Xop = "".join(xs)
                ops.append(Xop)
                coeffs.append(1 / np.sqrt(2))

        else:
            for i in range(self.nqubits - 1):
                zs = ["I"] * self.nqubits
                zs[i] = "Z"
                zs[i + 1] = "Z"
                Zop = "".join(zs)
                ops.append(Zop)
                coeffs.append(1.0 / np.sqrt(2))

                xs = ["I"] * self.nqubits
                xs[i] = "X"
                Xop = "".join(xs)
                ops.append(Xop)
                coeffs.append(1.0 / np.sqrt(2))

        return SparsePauliOp(ops, coeffs)

    def get_eigenvalues(self):
        hamiltonian = self.make_heis1()
        eigenvalues = (np.linalg.eigvals(hamiltonian.to_matrix())).real.tolist()
        eigenvalues.sort()
        E_max = eigenvalues[-1]
        E_min = eigenvalues[0]
        return E_max, E_min

    def ansatz_hamiltonian1(self, params):
        circuit = QuantumCircuit(self.nqubits)
        t = 0
        for i in range(self.nqubits):
            circuit.h(i)
        for idepth in range(self.depth):
            for i in range(self.nqubits):
                if i < (self.nqubits - 1):
                    j = i + 1
                else:
                    j = 0
                circuit.rzz(params[t], i, j)
                t += 1
            circuit.barrier()
            for i in range(self.nqubits):
                if i < (self.nqubits - 1):
                    j = i + 1
                else:
                    j = 0
                circuit.rx(params[t], i)
                t += 1
        return circuit

    def ansatz_hea(self, params):
        circuit = QuantumCircuit(self.nqubits)
        t = 0
        for idepth in range(self.depth):
            for i in range(self.nqubits):
                circuit.ry(params[t], i)
                t += 1
            circuit.barrier()
            for i in range(self.nqubits - 1):
                circuit.cx(i, i + 1)
        return circuit

    def cost_function1(self, params):
        hamiltonian = self.make_heis1()
        circuit = self.ansatz_hamiltonian1(params)
        circuit = circuit.decompose()
        noise_model = NoiseModel()
        error = depolarizing_error(self.error_rate, 1)
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
        error1 = depolarizing_error(self.error_rate * 10, 2)
        noise_model.add_all_qubit_quantum_error(error1, 'cx')
        sim_d = AerSimulator(noise_model=noise_model)
        circuit.save_statevector()
        circ_noise = transpile(circuit, sim_d)
        noise_result = sim_d.run(circ_noise, shots=1).result()
        u = noise_result.data(0)['statevector'].data
        expectation = (u.conj().dot(hamiltonian.to_matrix())).dot(u)
        return expectation.real

    def cost_function2(self, params):
        hamiltonian = self.make_heis1()
        circuit = self.ansatz_hea(params)
        circuit = circuit.decompose()
        noise_model = NoiseModel()
        error = depolarizing_error(self.error_rate, 1)
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
        error1 = depolarizing_error(self.error_rate * 10, 2)
        noise_model.add_all_qubit_quantum_error(error1, 'cx')
        sim_d = AerSimulator(noise_model=noise_model)
        circuit.save_statevector()
        circ_noise = transpile(circuit, sim_d)
        noise_result = sim_d.run(circ_noise, shots=1).result()
        u = noise_result.data(0)['statevector'].data
        expectation = (u.conj().dot(hamiltonian.to_matrix())).dot(u)
        return expectation.real

    def callback1(self, intermediate_result):
        # print(
        #    f"{intermediate_result.fun}"
        #    )
        with open('intermediate_values_hva.txt', 'a') as file:
            file.write(f'Intermediate values: {intermediate_result.fun}\n')

    def callback2(self, intermediate_result):
        # print(
        #    f"{intermediate_result.fun}"
        #    )
        with open('intermediate_values_hea.txt', 'a') as file:
            file.write(f'Intermediate values: {intermediate_result.fun}\n')

    def vqe(self):
        estimate_val = minimize(self.cost_function1, self.parameter, method="BFGS", tol=1e-5, options={'disp': False},
                                callback=self.callback1)
        return estimate_val

    def vqe2(self):
        estimate_val = minimize(self.cost_function2, self.parameter, method="BFGS", tol=1e-5, options={'disp': False},
                                callback=self.callback2)
        return estimate_val


## pure Cousine filtering
class CosineRunner:
    def __init__(self, nqubits, error_rate, time_step, periodic):
        self.nqubits = nqubits
        self.error_rate = error_rate
        self.time_step = time_step
        self.periodic = periodic

    def make_heis1(self):
        ops = []
        coeffs = []
        if self.periodic == True:
            for i in range(self.nqubits):
                if i < (self.nqubits - 1):
                    j = i + 1
                else:
                    j = 0
                zs = ["I"] * self.nqubits
                zs[i] = "Z"
                zs[j] = "Z"
                Zop = "".join(zs)
                ops.append(Zop)
                coeffs.append(1 / np.sqrt(2))

                xs = ["I"] * self.nqubits
                xs[i] = "X"
                # xs[i + 1] = "X"
                Xop = "".join(xs)
                ops.append(Xop)
                coeffs.append(1 / np.sqrt(2))

        else:
            for i in range(self.nqubits - 1):
                zs = ["I"] * self.nqubits
                zs[i] = "Z"
                zs[i + 1] = "Z"
                Zop = "".join(zs)
                ops.append(Zop)
                coeffs.append(1.0 / np.sqrt(2))

                xs = ["I"] * self.nqubits
                xs[i] = "X"
                Xop = "".join(xs)
                ops.append(Xop)
                coeffs.append(1.0 / np.sqrt(2))

        return SparsePauliOp(ops, coeffs)

    def depolarizing_channel(self, erate, density_matrix):
        E = (1 - erate) * density_matrix + erate * np.trace(density_matrix) * np.eye(
            2 ** self.nqubits) / 2 ** self.nqubits
        return E

    def get_eigenvalues(self):
        hamiltonian = self.make_heis1()
        eigenvalues = (np.linalg.eigvals(hamiltonian.to_matrix())).real.tolist()
        eigenvalues.sort()
        E_max = eigenvalues[-1]
        E_min = eigenvalues[0]
        return E_max, E_min

    def cosine_filtering(self, erate):
        hamiltonian = self.make_heis1()
        E_max, E_min = self.get_eigenvalues()
        # print(f'Ground state energy is {E_min}')
        hamiltonian_norm = hamiltonian.to_matrix() / E_max
        hamiltonian_matrix = hamiltonian_norm + 0.5 * self.nqubits * np.eye(2 ** self.nqubits)

        eigenvalues1 = np.linalg.eigvals(hamiltonian_matrix).real.tolist()
        eigenvalues1.sort()
        E_max = eigenvalues1[-1]
        dt = round(((np.pi / 2) / E_max), 1)
        circuit_temp = QuantumCircuit(self.nqubits)
        for i in range(self.nqubits):
            circuit_temp.ry(np.pi / 4, i)
        circuit_temp.save_statevector()
        simulator_temp = AerSimulator()
        circuit_temp_trans = transpile(circuit_temp, simulator_temp)
        v = simulator_temp.run(circuit_temp_trans).result().get_statevector()
        v = np.asarray(v)
        phi = np.outer(v, v.conj())
        U_cos = cosm(hamiltonian_matrix * dt)

        for i in range(self.time_step):
            phi = (U_cos.dot(phi)).dot(U_cos.conj())
            depolarizing = self.depolarizing_channel(erate, phi)
            phi = depolarizing
            phi = phi / np.trace(phi)
        expectation = np.trace(phi.dot(hamiltonian.to_matrix()))
        return expectation

    def cosine(self):
        energy_difference_list = []
        ground_state = self.get_eigenvalues()[1]
        for error in self.error_rate:
            expectation = self.cosine_filtering(erate=error)
            energy_difference = (expectation - ground_state) / np.abs(ground_state)
            energy_difference_list.append(energy_difference)

        plt.xticks(self.error_rate[:5], fontsize=5)
        plt.scatter(self.error_rate[:5], energy_difference_list[:5])
        plt.xscale('log')
        # Add any additional plotting code here
        noiseless = (energy_difference_list[5]) / np.abs(ground_state)
        plt.axhline(noiseless, 0, 1, color='r', linestyle='--', label='Noiseless')
        plt.legend()
        plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])
        plt.xlabel("error rate")
        plt.ylabel("Energy difference")
        plt.title(f'Unitary on {self.nqubits} qubits ZZ+X')
        plt.savefig(f'{self.nqubits}_cosine.png', dpi=300)
        plt.show()

        return energy_difference_list


### Experiments
def main():
    error_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0]
    time_step = 200
    if sys.argv[1] == "cosine":
        nqubits = int(sys.argv[2])
        runner = CosineRunner(nqubits=nqubits,
                              error_rate=error_rate,
                              time_step=time_step,
                              periodic=True)
        result = runner.cosine()
        # print(result)
    elif sys.argv[1] == "vqe":
        nqubits = int(sys.argv[2])
        depth = int(sys.argv[3])
        vqe_0 = VQE_Runner(nqubits=nqubits,
                           error_rate=0,
                           depth=depth,
                           periodic=True,
                           parameter=np.pi * np.random.random(2 * nqubits * depth))
        # vqe_0.ansatz_hamiltonian1(params=ParameterVector("a", length=8)).draw('mpl')
        vqe_hva = vqe_0.vqe()
        vqe_hea = vqe_0.vqe2()
        vqe_exact = vqe_0.get_eigenvalues()[1]
        # print(vqe_exact)
        # Store the final result
        with open('final_result.txt', 'w') as file:
            file.write(f'Exact values: {vqe_exact}\n')
            file.write(f'Final result HVA: {vqe_hva}\n')
            file.write(f'Final result HEA: {vqe_hea}\n')


if __name__ == "__main__":
    main()
