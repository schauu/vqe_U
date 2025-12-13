import pennylane as qml
import numpy as np



def reflection_matrix(main_wires, aux_wires):
    '''
    Builds the reflection matrix needed for qubitization.
    :param main_wires: main register
    :param aux_wires: auxiliary register
    :return: reflection matrix
    '''
    state = np.zeros(len(aux_wires))
    matrix = qml.matrix(2*qml.Projector(state, aux_wires)@qml.Identity(main_wires)- qml.Identity(aux_wires+main_wires))

    return matrix

dev = qml.device('default.qubit')

@qml.qnode(dev)
def my_qubitization(main_wires, aux_wires, state, hamiltonian):
    '''
    use the reflection matrix and qml.PrepSelPrep to construct Quantum walk operator Q
    :param main_wires:
    :param aux_wires:
    :param state: initial state for main register
    :param hamiltonian:
    :return: the output quantum state
    '''
    qml.StatePrep(state, wires=main_wires)
    qml.QubitUnitary(reflection_matrix(main_wires, aux_wires), wires=aux_wires+main_wires)
    qml.PrepSelPrep(hamiltonian, control=aux_wires)

    return qml.state()

# Now extract the phase by QPE
coef = [1/2, 1/8, 1/4]
ops = [qml.Z(0)@qml.Z(1), qml.Z(0), qml.Z(1)]
H = qml.ops.LinearCombination(coef, ops)

control_wires = [2,3]
estimation_wires = [4, 5, 6, 7, 8, 9, 10, 11, 12]

@qml.qnode(dev)
def PSP_QPE(state):
    '''
    Applies QPE to PSP encoding for H
    :param state: initial ground state candidate
    :return: the output probabilities
    '''
    qml.StatePrep(state, [0, 1]) #main register
    qml.QuantumPhaseEstimation(
        #qml.PrepSelPrep(H, control=control_wires), estimation_wires=estimation_wires
        qml.Qubitization(H, control=control_wires), estimation_wires=estimation_wires
    )
    return qml.probs(wires=estimation_wires)

results = [PSP_QPE(state) for state in np.eye(4)]
lambda_ = sum(abs(x) for x in coef)
#theta_k = [np.pi *2 * np.argmax(results)/ 2**len(estimation_wires) for result in results]
#eigenvalues = lambda_ * np.cos(theta_k)
eigenvalues = [lambda_ * np.cos(2 * np.pi * np.argmax(result) / 2 ** (len(estimation_wires)))
               for result in results]
print('E = ', eigenvalues)

print('reference is', np.linalg.eigvals(qml.matrix(H)))