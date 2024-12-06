#!/usr/bin/env python3
import numpy as np
import scipy
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity, Operator
from qiskit.synthesis import SuzukiTrotter, LieTrotter
from qiskit_aer import AerSimulator
from qiskit_algorithms import TimeEvolutionProblem, TrotterQRTE
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.primitives import Estimator
from qiskit.circuit.library import PauliEvolutionGate
import matplotlib.pyplot as plt
from scipy.linalg import expm, cosm
import warnings
import sys
import functools
import scipy as sc
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize


# warnings.simplefilter("ignore", np.ComplexWarning)
# np.set_printoptions(linewidth=100)

def get_hamiltonian(nqubits):
    J = 1 / np.sqrt(2)
    ZZ_tuples = [('ZZ', [i, i + 1], J) for i in range(nqubits - 1)]
    ZZ_tuples += [('ZZ', [nqubits - 1, 0], J)]
    X_tuples = [("X", [i], J) for i in range(nqubits)]
    hamiltonian = SparsePauliOp.from_sparse_list([*ZZ_tuples, *X_tuples], num_qubits=nqubits)
    return hamiltonian.simplify()


def ansatz_hamiltonian(nqubits, depth, params):
    circuit = QuantumCircuit(nqubits)
    t = 0
    for i in range(nqubits):
        circuit.h(i)
    for idepth in range(depth):
        for i in range(nqubits):
            if i < (nqubits - 1):
                j = i + 1
            else:
                j = 0
            circuit.rzz(params[t], i, j)
            t += 1
        circuit.barrier()
        for i in range(nqubits):
            if i < (nqubits - 1):
                j = i + 1
            else:
                j = 0
            circuit.rx(params[t], i)
            t += 1
    return circuit


def ansatz_hea(nqubits, depth, params):
    circuit = QuantumCircuit(nqubits)
    t = 0
    for idepth in range(depth):
        for i in range(nqubits):
            circuit.ry(params[t], i)
            t += 1
        circuit.barrier()
        for i in range(nqubits - 1):
            circuit.cx(i, i + 1)
    return circuit


def cost_function1(params, nqubits, depth, error_rate):
    hamiltonian = get_hamiltonian(nqubits)
    circuit = ansatz_hamiltonian(nqubits, depth, params)
    circuit = circuit.decompose()
    noise_model = NoiseModel()
    error = depolarizing_error(error_rate, 1)
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
    error1 = depolarizing_error(error_rate * 10, 2)
    noise_model.add_all_qubit_quantum_error(error1, 'cx')
    sim_d = AerSimulator(noise_model=noise_model)
    circuit.save_statevector()
    if error_rate == 0:
        simulator = AerSimulator()
    else:
        simulator = sim_d
        circuit = transpile(circuit, sim_d)
        # noise_result = sim_d.run(circ_noise, shots=1).result()
    result = simulator.run(circuit).result()
    u = result.data(0)['statevector'].data
    expectation = (u.conj().dot(hamiltonian.to_matrix())).dot(u)
    return expectation.real


def cost_function2(params, nqubits, depth, error_rate):
    hamiltonian = get_hamiltonian(nqubits)
    circuit = ansatz_hea(nqubits, depth, params)
    circuit = circuit.decompose()
    noise_model = NoiseModel()
    error = depolarizing_error(error_rate, 1)
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
    error1 = depolarizing_error(error_rate * 10, 2)
    noise_model.add_all_qubit_quantum_error(error1, ['cx'])
    sim_d = AerSimulator(noise_model=noise_model)
    circuit.save_statevector()
    if error_rate == 0:
        simulator = AerSimulator()
    else:
        simulator = sim_d
        circuit = transpile(circuit, sim_d)
        # noise_result = sim_d.run(circ_noise, shots=1).result()
    result = simulator.run(circuit).result()
    u = result.data(0)['statevector'].data
    expectation = (u.conj().dot(hamiltonian.to_matrix())).dot(u)
    return expectation.real


def callback1(intermediate_result):
    # print(
    #    f"{intermediate_result.fun}"
    #    )
    with open('intermediate_values_hva.txt', 'a') as file:
        file.write(f'Intermediate values: {intermediate_result.fun}\n')


def callback2(intermediate_result):
    # print(
    #    f"{intermediate_result.fun}"
    #    )
    with open('intermediate_values_hea.txt', 'a') as file:
        file.write(f'Intermediate values: {intermediate_result.fun}\n')


## pure Cousine filtering
## 1st order
def construct_trotter_circuit(H, time, nqubits, order, time_step):
    if order == 1:
        formular = LieTrotter(reps=time_step)
    else:
        formular = SuzukiTrotter(order=order, reps=time_step)
    trotter_step_first_order = PauliEvolutionGate(H, time, synthesis=formular)
    circuit = QuantumCircuit(nqubits+1)
    circuit.append(trotter_step_first_order, range(nqubits+1))
    #circuit = circuit.decompose(reps=2)
    return circuit

def select_sim(error_rate):
    noise_model = NoiseModel()
    error = depolarizing_error(error_rate, 1)
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
    error1 = depolarizing_error(error_rate*10, 2)
    noise_model.add_all_qubit_quantum_error(error1,'cx')
    sim_d = AerSimulator(noise_model=noise_model)
    if error_rate==0:
        simulator = AerSimulator()
    else:
        simulator = sim_d
    return simulator

def unitary_exact(time, nqubits, error_rate, step):
    projector = np.array([[0, -1.0j],
                     [1.0j, 0]])
    J = 1/np.sqrt(2)
    hamiltonian = get_hamiltonian(nqubits)
    H_array = hamiltonian.to_matrix()
    eval, _ = np.linalg.eigh(H_array)
# print(eval)
    emin = eval[0]
    emax = eval[-1]
    H_array = (H_array - emin * np.eye(2**nqubits)) / (emax - emin)
    initial_state = Statevector.from_label("0"*nqubits)
    expectation_exact = []
    simulator = select_sim(error_rate)
    statevector = initial_state.data
    projector2 = np.array([[1, 0],
                           [0, 0]])  ## for expectation
    hamiltonian1 = np.kron(projector2, get_hamiltonian(nqubits).to_matrix())
    hamiltonian1 = Operator(hamiltonian1)
    hamiltonian1 = SparsePauliOp.from_operator(hamiltonian1)
    hamiltonian2 = np.kron(projector2, np.eye(2 ** nqubits))
    for i in range(step):
        U = expm(-1.0j * time * np.kron(projector, H_array))
        qc = QuantumCircuit(nqubits+1)
        qc.initialize(statevector, range(nqubits))
        qc.unitary(U, range(nqubits+1))
        qc.decompose(reps=2)
        qc.save_statevector()
        result = simulator.run(qc).result().data(0)['statevector']
        new_state = result.data#[:2**nqubits]
        expectation = (new_state.conj().dot(hamiltonian1.to_matrix())).dot(new_state)/(new_state.conj().T.dot(hamiltonian2).dot(new_state))
        expectation_exact.append(expectation)
        statevector = new_state[:2**nqubits]/np.linalg.norm(new_state[:2**nqubits])
    return expectation_exact

def unitary_trotter(H, time, nqubits, order, time_step, error_rate, step):
    simulator = select_sim(error_rate)
    expectation_list = []
    initial_state = Statevector.from_label("0" * nqubits)
    statevector = initial_state.data
    # H = get_hamiltonian_y(5, 1/np.sqrt(2), True)
    projector = np.array([[0, -1.0j],
                          [1.0j, 0]])

    projector2 = np.array([[1, 0],
                           [0, 0]])  ## for expectation
    hamiltonian1 = np.kron(projector2, get_hamiltonian(nqubits).to_matrix())
    hamiltonian1 = Operator(hamiltonian1)
    hamiltonian1 = SparsePauliOp.from_operator(hamiltonian1)
    hamiltonian2 = np.kron(projector2, np.eye(2 ** nqubits))
    for i in range(step):
        qc = QuantumCircuit(nqubits+1)
        qc.initialize(statevector, range(nqubits))
        circuit_temp = construct_trotter_circuit(H, time, nqubits, order, time_step)
        qc = qc.compose(circuit_temp, range(nqubits+1))
        #qc = qc.decompose(reps=2)
        qc.save_statevector()
        circuit = transpile(qc, simulator)
        result = simulator.run(circuit).result().data(0)['statevector']
        new_state = result.data#[:2**nqubits]
        # Print the statevector at each step for debugging
        #print(f"Step {i}: Statevector = {statevector}")
        expectation = (new_state.conj().T.dot(hamiltonian1.to_matrix())).dot(new_state)/(new_state.conj().T.dot(hamiltonian2).dot(new_state))
        expectation_list.append(expectation)
        # Print the expectation value at each step for debugging
        #print(f"Step {i}: Expectation = {expectation}")
        statevector = new_state[:2**nqubits]/np.linalg.norm(new_state[:2**nqubits])
    return expectation_list

### Experiments
def main():
    np.random.seed(42)
    if sys.argv[1] == "cosine":
        error_rate = int(float(sys.argv[3]))
        nqubits = int(sys.argv[2])
        J = 1 / np.sqrt(2)
        hamiltonian = get_hamiltonian(nqubits)
        H_array = hamiltonian.to_matrix()
        eval, _ = np.linalg.eigh(H_array)
        emin = eval[0]
        emax = eval[-1]
        H_array = (H_array - emin * np.eye(2 ** nqubits)) / (emax - emin)  ## scale H
        final_time = np.pi / 2  # time
        time = final_time
        time_step = 3  # time_step
        projector = np.array([[0, -1.0j],
                              [1.0j, 0]])
        H = Operator(np.kron(projector, H_array))
        H = SparsePauliOp.from_operator(H)


        step = 40
        order = 1
        expectation_1 = unitary_trotter(H, time, nqubits, order, time_step, error_rate, step)
        order = 2
        expectation_2 = unitary_trotter(H, time, nqubits, order, time_step, error_rate, step)
        order = 4
        expectation_4 = unitary_trotter(H, time, nqubits, order, time_step, error_rate, step)

        expectation_exact = unitary_exact(time, nqubits, error_rate, step)

        fig, axes = plt.subplots()
        x = list(range(1, step + 1))  # includes initial state
        axes.plot(
            x, expectation_1, label="First order", marker="x", c="darkmagenta", ls="-", lw=0.8
        )
        axes.plot(
            x, expectation_2, label="Second order", marker="o", c="limegreen", ls="-", lw=0.8
        )
        axes.plot(
            x, expectation_4, label="Fourth order", marker="v", c="r", ls="-", lw=0.8
        )
        axes.plot(x, expectation_exact, c="k", ls=":", label="Exact matrix exponential")
        horizontal_line_value = emin
        axes.axhline(y=horizontal_line_value, color='r', linestyle='--',
                     label='minimum eigenvalue')  # f'y = {horizontal_line_value}')
        legend = fig.legend(
            *axes.get_legend_handles_labels(),
            bbox_to_anchor=(1.0, 0.5),
            loc="center left",
            framealpha=0.5,
        )

        axes.set_xticks(np.arange(0, max(x) + 1, 5))
        axes.set_xlabel('Step')
        axes.set_ylabel('Energy')
        axes.set_title('Trotter step is 3')
        fig.tight_layout()
        fig.savefig("trotter_step3.png", dpi=300)

        # print(result)
    elif sys.argv[1] == "vqe":
        error_rate = 0
        nqubits = int(sys.argv[2])
        depth = int(sys.argv[3])
        parameter = np.array(np.random.random(2 * nqubits * depth))
        circuit = ansatz_hamiltonian(nqubits, depth, parameter)
        circuit = circuit.decompose(reps=2)
        count = []
        count.append(circuit.depth())  # depth, total gate, nonlocal gates
        count.append(len(circuit))
        count.append(circuit.num_nonlocal_gates())
        print(count)
        estimate_val_hva = minimize(cost_function1, parameter, args=(nqubits, depth, error_rate), method="BFGS",
                                    tol=1e-5, options={'disp': False}, callback=callback1)
        estimate_val_hea = minimize(cost_function2, parameter, args=(nqubits, depth, error_rate), method="BFGS",
                                    tol=1e-5, options={'disp': False}, callback=callback2)
        # print(vqe_exact)
        # Store the final result
        with open('final_result.txt', 'w') as file:
            #file.write(f'Exact values: {vqe_exact}\n')
            file.write(f'Final result HVA: {estimate_val_hva}\n')
            file.write(f'Final result HEA: {estimate_val_hea}\n')


if __name__ == "__main__":
    main()
