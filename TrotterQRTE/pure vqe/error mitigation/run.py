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


def ansatz_vqe(nqubits, depth, params):
    circuit = QuantumCircuit(nqubits + 1)
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
        circuit.ry(params[t], nqubits)
        t += 1
    return circuit


def ansatz_hamiltonian(nqubits, depth, params):
    circuit = QuantumCircuit(nqubits)
    t = 0
    for i in range(nqubits):
        circuit.h(i)
    for _ in range(depth):
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
    for _ in range(depth):
        for i in range(nqubits):
            circuit.ry(params[t], i)
            t += 1
        circuit.barrier()
        for i in range(nqubits - 1):
            circuit.cx(i, i + 1)
    return circuit

def ansatz_1(nqubits, depth, params):
    circuit = QuantumCircuit(nqubits)
    t = 0
    for idepth in range(depth):
        for i in range(nqubits):
            circuit.ry(params[t], i)
            t += 1
        circuit.barrier()
        for i in range(nqubits-1):
            circuit.cry(params[t], i, i+1)
            t+=1
    return circuit
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


class VQECallback:
    def __init__(self, sim, circuit, nqubits, reference, H_matrix, threshold, param):
        self.sim = sim
        self.circuit = circuit
        self.nqubits = nqubits
        self.reference = reference
        self.H_matrix = H_matrix
        self.threshold = threshold
        self.param = param



    def run_sim(self, params):
        new_circuit = self.circuit.assign_parameters(params)
        result = self.sim.run(new_circuit).result()
        statevector = result.data(0)["statevector"].data
        return statevector

    def cost_func(self, params):
        statevector = self.run_sim(params)
        probability = np.linalg.norm(statevector[: 2**self.nqubits])
        return -probability

    def cost_func_3(self, params):
        # for cosine_filtering_vqe_3
        statevector = self.run_sim(params)
        statevector = statevector[: 2**self.nqubits]
        probability = np.linalg.norm(statevector[: 2**self.nqubits])
        statevector /= probability
        expectation = statevector.conj().dot(self.H_matrix.dot(statevector)).real
        return -probability + 1 * (expectation - self.reference)

    def constraint_func(self, params):
        statevector = self.run_sim(params)
        statevector = statevector[: 2**self.nqubits]
        statevector /= np.linalg.norm(statevector)

        expectation = statevector.conj().T.dot(self.H_matrix.dot(statevector)).real
        return -(expectation - self.reference - self.threshold)
        #return -np.abs(expectation - self.reference - self.threshold)


def cosine_filtering_vqe(H, time, nqubits, order, time_step, error_rate, step, threshold):
    simulator = select_sim(error_rate)
    expectation_before_list = []
    probability_before_list = []
    expectation_after_list = []
    probability_after_list = []
    initial_state = Statevector.from_label("0000")
    statevector = initial_state.data

    H_cosine = SparsePauliOp(["Y"]) ^ H
    H_matrix = H.to_matrix()

    vqe_depth = 3
    vqe_nparams = vqe_depth *((nqubits+1)*2-1) ##hea+cry
    #vqe_nparams = vqe_depth * (2 * (nqubits + 1)) ## hva+ry
    # vqe_nparams = vqe_depth * 2 * nqubits  ## hva
    #vqe_nparams = vqe_depth * (nqubits+1) ## hea
    vqe_params = ParameterVector("θ", vqe_nparams)
    #parameter = np.random.random(vqe_nparams)

    for i in range(step):
        print(f"Running {i}", flush=True)
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
        expectation = postselect_state.conj().T.dot(H_matrix.dot(postselect_state)) ##参考
        if i == 0:
            probability = probability
            expectation = expectation
        else:
            probability *= probability_before_list[-1]
            expectation = expectation_after_list[-1]

        probability_before_list.append(probability)
        expectation_before_list.append(expectation)
        statevector = postselect_state
        ## vqe part
        result_intermediate = []

        def callback(intermediate_result):
            statevector = vqe_callback.run_sim(intermediate_result)
            probability = np.linalg.norm(statevector[: 2 ** nqubits])
            result_intermediate.append(probability)
            #result_intermediate.append(intermediate_result.fun)


        vqe_qc = qc1
        vqe_ansatz = ansatz_1(nqubits+1, vqe_depth, vqe_params)
        vqe_qc.compose(vqe_ansatz, range(nqubits + 1), inplace=True)
        vqe_qc.save_statevector()
        vqe_qc = transpile(vqe_qc, simulator)
        vqe_callback = VQECallback(
            simulator, vqe_qc, nqubits, expectation.real, H_matrix, -1e-2, vqe_params
        )

        np.random.seed(42)
        parameter = np.random.random(vqe_nparams)
        # parameter = np.zeros(vqe_nparams, dtype=np.float64)
        vqe_result = sp.optimize.minimize(
            vqe_callback.cost_func,
            parameter,
            method="SLSQP",
            constraints={"type": "ineq", "fun": vqe_callback.constraint_func},
            options={"maxiter" : 1000},
            callback=callback,
        )
        print(vqe_result)
        #parameter = vqe_result.x
        postselect_state = vqe_callback.run_sim(vqe_result.x)[: 2**nqubits]
        probability = np.linalg.norm(postselect_state)
        print(f"Probability at this step is {probability}")
        postselect_state /= probability
        expectation = postselect_state.conj().T.dot(H_matrix.dot(postselect_state))
        if expectation <= threshold:
            break
        if i == 0:
            probability = probability
        else:
            probability *= probability_after_list[-1]
        print(f"Probability consider previous step is {probability}")
        probability_after_list.append(probability)
        expectation_after_list.append(expectation)
        statevector = postselect_state

    expectation_before_list = np.array(expectation_before_list)
    probability_before_list = np.array(probability_before_list)
    expectation_after_list = np.array(expectation_after_list)
    probability_after_list = np.array(probability_after_list)

    assert np.max(np.abs(np.imag(expectation_before_list))) < 1e-6
    assert np.max(np.abs(np.imag(expectation_after_list))) < 1e-6

    return (
        expectation_before_list.real,
        probability_before_list,
        expectation_after_list.real,
        probability_after_list,
        result_intermediate,
    )


def cost_function(params, nqubits, depth, error_rate):
    J = 1/np.sqrt(2)
    hamiltonian =  get_hamiltonian(nqubits, J)
    circuit =  ansatz_hamiltonian(nqubits, depth,params)
    circuit = circuit.decompose()
    noise_model = NoiseModel()
    error = depolarizing_error(error_rate, 1)
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
    error1 = depolarizing_error(error_rate*10, 2)
    noise_model.add_all_qubit_quantum_error(error1,'cx')
    #circuit.save_statevector()
    if error_rate == 0:
        estimator = EstimatorV2()
        #print('error rate is 0')
    else:
        #print('error rate is ' + str(error_rate))
        estimator = EstimatorV2(options=dict(backend_options=dict(noise_model=noise_model)))
        #circuit = transpile(circuit, sim_d)
        #noise_result = sim_d.run(circ_noise, shots=1).result()
    result =estimator.run([(circuit, hamiltonian)]).result()
    expectation=result[0].data.evs
    return expectation.real

def vqe(nqubits, depth, error_rate):
    vqe_nparams = depth * 2 * nqubits  ## hva
    #vqe_nparams = depth * nqubits ## hea
    vqe_params = np.random.random(vqe_nparams)
    result_intermediate = []
    def callback(intermediate_result):
        result_intermediate.append(intermediate_result.fun)
    estimate_val = sp.optimize.minimize(cost_function, vqe_params, args=(nqubits, depth, error_rate), method="BFGS",
                                        tol=1e-5,
                                        options={'disp': True}, callback=callback)
    print(estimate_val)
    return result_intermediate

def plot_data3(
    filename, nstep, intermediate_values):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    step = list(range(nstep[0]))
    step1 = list(range(nstep[1]))
    step2 = list(range(nstep[2]))
    step3 = list(range(nstep[3]))
    colors = ["darkmagenta", "limegreen", "k","r"]
    markers = ["*", "x", "v", "1"]
    ax1.plot(
        step,
        intermediate_values[0],
        label="Expect error 0",
        #marker=markers[0],
        c=colors[0],
        ls="-",
        lw=0.8,
    )
    ax1.plot(
        step1,
        intermediate_values[1],
        label="Expect error 1e-4",
        #marker=markers[1],
        c=colors[1],
        ls="-",
        lw=0.8,
    )
    ax1.plot(
        step2,
        intermediate_values[2],
        label="Expect error 1e-3",
        #marker=markers[2],
        c=colors[2],
        ls="-",
        lw=0.8,
    )
    ax1.plot(
        step3,
        intermediate_values[3],
        label="Expect error 1e-2",
        #marker=markers[3],
        c=colors[3],
        ls="-",
        lw=0.8,
    )
    ax1.axhline(
        y=np.min(np.concatenate(intermediate_values)),
        color="r",
        linestyle="--",
        label="Minimum expectation",
    )
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Expectation", color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])

    handles1, labels1 = ax1.get_legend_handles_labels()

    all_handles = handles1
    all_labels = labels1

    legend = fig.legend(
        all_handles,
        all_labels,
        bbox_to_anchor=(0, 0, 1, 1),
        loc="upper right",
        # ncol=3,
        borderaxespad=0,
        fontsize=8,
        bbox_transform=ax1.transAxes
    )

    #ax1.set_title("Only VQE for expectation over Steps")

    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    fig.savefig(filename, dpi=600)

if __name__ == "__main__":
    nqubits = 8
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
    error_rate1 = 1e-4
    error_rate2 = 1e-3
    error_rate3 = 1e-2
    order = 2
    threshold = 1e-6

    H = SparsePauliOp.from_operator(Operator(H_array))

    depth = nqubits - 1
    ##pure VQE
    #cut_off = 30
    result = vqe(nqubits, depth, error_rate)
    #result = result[:cut_off]
    result1 = vqe(nqubits, depth, error_rate1)
    #result1 = result1[:cut_off]
    result2 = vqe(nqubits, depth, error_rate2)
    #result2 = result2[:cut_off]
    result3 = vqe(nqubits, depth, error_rate3)
    #result3 = result3[:cut_off]
    nstep = [len(result), len(result1), len(result2), len(result3)]
    result_intermediate = [result, result1, result2, result3]
    plot_data3('pure VQE hva 1', nstep, result_intermediate)
    with open('pure_vqe_hva.txt', 'w', encoding='utf-8') as f:
        for i in range(len(result)):
            f.write(f"{result[i]}\t{result1[i]}\t{result2[i]}\t{result3[i]}\n")
        print('Result has been stored')

