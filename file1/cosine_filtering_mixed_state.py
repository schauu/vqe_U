#!/usr/bin/env python3

import numpy as np
import scipy as sp
from qiskit.quantum_info import SparsePauliOp

np.set_printoptions(linewidth=1000000)


def postselect_rho(density_matrix, postselect_qubit, postselect_value):
    nq = int(np.log2(density_matrix.shape[0]))
    assert postselect_qubit < nq

    indices = np.arange(1 << nq)
    mask = (indices & (1 << postselect_qubit)).astype(bool)
    if not postselect_value:
        mask = np.logical_not(mask)
    indices = indices[mask]

    return density_matrix[np.ix_(indices, indices)]


def cosine_filtering(initial_state, scaled_hamiltonian, dt, nreps):
    Y = np.array([[0, -1j], [1j, 0]])
    U = sp.linalg.expm(-1j * np.kron(Y, scaled_hamiltonian) * dt)

    nstates = initial_state.shape[0]
    nq = int(np.log2(nstates))

    rho = initial_state
    for i in range(nreps):
        rho_1 = np.pad(rho, (0, nstates))
        Urho_1 = U @ rho_1
        new_rho = postselect_rho(Urho_1, nq, 0)
        new_rho /= np.trace(new_rho)
        exp = np.einsum("ij,ji->", new_rho, scaled_hamiltonian)
        rho = new_rho
        print(f"Step #{i}: {exp}")


def main():
    nq = 4
    hamiltonian = SparsePauliOp(
        ["IIZZ", "IZZI", "ZZII", "ZIIZ", "IIIX", "IIXI", "IXII", "XIII"]
    )
    hmat = hamiltonian.to_matrix()
    eval, _ = np.linalg.eigh(hmat)
    # print(eval)
    emin = eval[0]
    emax = eval[-1]
    hmat = (hmat - emin * np.eye(2**nq)) / (emax - emin)
    dt = np.pi / 2

    eval, _ = np.linalg.eigh(hmat)

    initial_state = 0
    # initial_state option 1
    temp = sp.linalg.hadamard(1 << nq, dtype=np.float64)
    for i in range(1 << nq):
        state = temp[:, i]
        initial_state += np.outer(state, state)

    # initial_state option 2
    # for i in range(1 << nq):
    #     state = np.random.random(1 << nq)
    #     initial_state += np.outer(state, state)

    initial_state /= np.trace(initial_state)

    cosine_filtering(initial_state, hmat, dt, 1000)


if __name__ == "__main__":
    main()
