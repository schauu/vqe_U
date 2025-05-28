#!/usr/bin/env python3


import matplotlib.pyplot as plt
import math
import pennylane as qml
from pennylane import numpy as np
from scipy.linalg import expm

# Step 1: Construct TFIM matrix H
n = 4  # Data qubits (system qubits)
coeffs, ops = [], []
for i in range(n - 1):
    coeffs.append(-1.0)
    ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
for i in range(n):
    coeffs.append(-1.0)
    ops.append(qml.PauliX(i))

H_tfim = qml.Hamiltonian(coeffs, ops)
H_mat = qml.matrix(H_tfim)
# Step 5: Exact Ground State Calculation using NumPy's linalg.eig
eigenvalues, eigenvectors = np.linalg.eig(H_mat)

min_index = np.argmin(eigenvalues)
min_eigenvalue = eigenvalues[min_index]
min_eigenvector = eigenvectors[:, min_index]

H_mat -= min_eigenvalue * np.eye(2**n)
# Step 2: LCU form of cos^{2m}(H)
m = 10
k_vals = np.arange(-m, m + 1)
alpha_k = 2**(-2 * m) * np.array([math.comb(2 * m, m + k) for k in k_vals])
U_k = [expm(2j * k * H_mat) for k in k_vals]
U_k_ops = [qml.Hermitian(U.real, wires=range(n)) for U in U_k]  # use Hermitian(real) approximation
cos2m_op = qml.dot(alpha_k, U_k_ops)

# Step 3: Define Qubitization circuit
n_control = int(np.ceil(np.log2(len(alpha_k))))
control_wires = [n + i for i in range(n_control)]  # control wires for ancilla qubits
aux_wire = n + n_control

# Define a device with n + 1 qubits (4 data qubits + 1 ancilla qubit)
dev = qml.device("default.qubit", wires=n + n_control)

@qml.qnode(dev)
def circuit():
    # Apply Qubitization to create the state
    qml.Qubitization(cos2m_op, control=control_wires)

    # Return the final state vector
    return qml.state()

# Step 4: Run the circuit and get the state vector
state_vector = circuit()[:2**n]
state_vector/=np.linalg.norm(state_vector)
# Verify the shape of the state vector (should be 16, 1)


circuit_diagram = qml.draw_mpl(circuit, style='pennylane')
print(circuit_diagram())


# Print results
# print("Exact ground state (via NumPy):", min_eigenvector)
# print("Qubitization state vector:", state_vector)
state0 = qml.math.dm_from_state_vector(min_eigenvector)
state1 = qml.math.dm_from_state_vector(state_vector)
print('fidelity is :', qml.math.fidelity(state0, state1))
