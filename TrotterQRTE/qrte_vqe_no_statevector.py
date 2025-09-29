#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import Estimator
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, SuzukiTrotter, MatrixExponential
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_algorithms import TimeEvolutionProblem, TrotterQRTE

#from TrotterQRTE.pure_VQE import result_intermediate


def get_hamiltonian(nq, J):
    op_list = [("ZZ", [i, i + 1], J) for i in range(nq - 1)]
    op_list.append(("ZZ", [nq - 1, 0], J))
    op_list += [("X", [i], J) for i in range(nq)]
    return SparsePauliOp.from_sparse_list(op_list, num_qubits=nq).simplify()


def construct_trotter_circuit(H, time, nqubits, order, time_step):
    if order == 0:
        formular = MatrixExponential()
    elif order == 1:
        formular = LieTrotter(reps=time_step)
    else:
        formular = SuzukiTrotter(order=order, reps=time_step)
    trotter_step_first_order = PauliEvolutionGate(H, time, synthesis=formular)
    circuit = QuantumCircuit(nqubits + 1)
    circuit.append(trotter_step_first_order, range(nqubits + 1))
    # circuit = circuit.decompose(reps=2)
    return circuit


def select_sim(error_rate):
    noise_model = NoiseModel()
    error = depolarizing_error(error_rate, 1)
    noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"])
    error1 = depolarizing_error(error_rate * 10, 2)
    noise_model.add_all_qubit_quantum_error(error1, "cx")
    sim_d = AerSimulator(noise_model=noise_model)
    if error_rate == 0:
        simulator = AerSimulator()
    else:
        simulator = sim_d
    return simulator


def cosine_filtering(H, time, nqubits, order, time_step, error_rate, step, threshold):
    simulator = select_sim(error_rate)
    expectation_list = []
    probability_list = []
    initial_state = Statevector.from_label("0000")
    statevector = initial_state.data

    H_cosine = SparsePauliOp(["Y"]) ^ H
    H_matrix = H.to_matrix()

    for i in range(step):
        ## cosine filter part
        qc = QuantumCircuit(nqubits + 1)
        qc.initialize(statevector, range(nqubits))
        circuit_temp = construct_trotter_circuit(
            H_cosine, time, nqubits, order, time_step
        )
        qc = qc.compose(circuit_temp, range(nqubits + 1))
        qc1 = qc.copy()
        qc.save_statevector()
        circuit = transpile(qc, simulator)
        result = simulator.run(circuit).result().data(0)["statevector"]
        new_state = result.data
        postselect_state = new_state[: 2**nqubits]
        probability = np.linalg.norm(postselect_state)
        postselect_state /= probability
        expectation = postselect_state.conj().T.dot(H_matrix.dot(postselect_state))
        if expectation <= threshold:
            break
        if i == 0:
            probability = probability
        else:
            probability *= probability_list[-1]
        probability_list.append(probability)
        expectation_list.append(expectation)
        statevector = postselect_state

    expectation_list = np.array(expectation_list)
    probability_list = np.array(probability_list)

    assert np.max(np.abs(np.imag(expectation_list))) < 1e-6

    return expectation_list.real, probability_list

if __name__ == "__main__":
    nqubits = 4
    J = 1.0 / np.sqrt(2)
    hamiltonian = get_hamiltonian(nqubits, J)
    H_array = hamiltonian.to_matrix()
    eval, _ = np.linalg.eigh(H_array)
    # print(eval)
    emin = eval[0]
    emax = eval[-1]
    H_array = (H_array - emin * np.eye(2**nqubits)) / (emax - emin)  ## scale H
    final_time = np.pi / 2  # time
    time = final_time
    time_step = 3  # trotter step
    initial_state = Statevector.from_label("00000")
    error_rate = 0
    order = 2
    nstep = 25
    threshold = 1e-6
    H = SparsePauliOp.from_operator(Operator(H_array))
    depth = 3
    order = 2
    f_e2, f_p2 = cosine_filtering(H, time, nqubits, order, time_step, error_rate, nstep, threshold)
    with open('test.txt', 'w', encoding='utf-8') as f:
      f.write(f"{f_e2}\t{f_p2}\n")
    print('Result has been stored')


