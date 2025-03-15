#!/usr/bin/env python3
import numpy as np


def postselect_statevector(
    statevector: np.ndarray,
    nqubits: int,
    select_qubit: int,
    select_value: int,
    normalize: bool = True,
    print_debug: bool = False,
) -> np.ndarray:
    """
    statevector: input statevector (the numpy array)
    nqubits: number of qubits
    select_qubit: qubit index to postselect
    select_value: 0 to post-select on |0>, 1 to post-select on |1>
    normalize: normalize result
    """
    if not isinstance(statevector, np.ndarray):
        statevector = statevector.data
    if 2**nqubits != len(statevector):
        raise ValueError("nqubit inconsistent with statevector length")
    if nqubits <= 1:
        raise ValueError(f"nqubits out of range: {nqubits}")
    if select_qubit < 0 or select_qubit >= nqubits:
        raise ValueError(f"select_qubit out of range: {select_qubit}")
    if select_value not in (0, 1):
        assert ValueError(f"select_value invalid value (expect 0 or 1): {select_value}")
    if nqubits >= 64:
        assert ValueError(f"nqubits too large, must be less than 64")
    if print_debug:
        print(
            f"Postselecting {nqubits}-qubit statevector selecting qubit #{select_qubit} to |{select_value}>"
        )

    mask1 = (1 << select_qubit) - 1
    mask2 = (1 << select_qubit) if select_value == 1 else 0
    mask3 = ((1 << (nqubits - 1)) - 1) ^ mask1
    if print_debug:
        print(f"Mask1: {mask1:0{nqubits}b}")
        print(f"Mask2: {mask2:0{nqubits}b}")
        print(f"Mask3: {mask3:0{nqubits-1}b} ")

    dst_indices = np.arange(2 ** (nqubits - 1), dtype=np.int64)
    # below is equivalent to (but using numpy arrays)
    # src_indices = (dst_indices & mask1) | mask2 | ((dst_indices & mask3) << 1)
    src_indices = np.bitwise_and(dst_indices, mask1)
    src_indices = np.bitwise_or(src_indices, mask2)
    src_indices = np.bitwise_or(
        src_indices, np.left_shift(np.bitwise_and(dst_indices, mask3), 1)
    )
    if print_debug:
        print("")
        for src_state, dst_state in zip(src_indices, dst_indices):
            print(f"{src_state:0{nqubits}b} => {dst_state:0{nqubits-1}b}")
    res = statevector[src_indices]
    if normalize:
        res /= np.linalg.norm(res)
    return res


def postselect_density_matrix(
    density_matrix: np.ndarray,
    nqubits: int,
    select_qubit: int,
    select_value: int,
    normalize: bool = True,
    print_debug: bool = True,
) -> np.ndarray:
    """
    density_matrix: input density_matrix (the numpy array)
    nqubits: number of qubits
    select_qubit: qubit index to postselect
    select_value: 0 to post-select on |0>, 1 to post-select on |1>
    normalize: normalize result
    """
    if not isinstance(density_matrix, np.ndarray):
        density_matrix = density_matrix.data
    if 2**nqubits != len(density_matrix):
        raise ValueError("nqubit inconsistent with density_matrix length")
    if nqubits <= 1:
        raise ValueError(f"nqubits out of range: {nqubits}")
    if select_qubit < 0 or select_qubit >= nqubits:
        raise ValueError(f"select_qubit out of range: {select_qubit}")
    if select_value not in (0, 1):
        assert ValueError(f"select_value invalid value (expect 0 or 1): {select_value}")
    if nqubits >= 64:
        assert ValueError(f"nqubits too large, must be less than 64")
    if print_debug:
        print(
            f"Postselecting {nqubits}-qubit density_matrix selecting qubit #{select_qubit} to |{select_value}>"
        )

    mask1 = (1 << select_qubit) - 1
    mask2 = (1 << select_qubit) if select_value == 1 else 0
    mask3 = ((1 << (nqubits - 1)) - 1) ^ mask1
    if print_debug:
        print(f"Mask1: {mask1:0{nqubits}b}")
        print(f"Mask2: {mask2:0{nqubits}b}")
        print(f"Mask3: {mask3:0{nqubits-1}b} ")

    dst_indices = np.arange(2 ** (nqubits - 1), dtype=np.int64)
    # below is equivalent to (but using numpy arrays)
    # src_indices = (dst_indices & mask1) | mask2 | ((dst_indices & mask3) << 1)
    src_indices = np.bitwise_and(dst_indices, mask1)
    src_indices = np.bitwise_or(src_indices, mask2)
    src_indices = np.bitwise_or(
        src_indices, np.left_shift(np.bitwise_and(dst_indices, mask3), 1)
    )
    if print_debug:
        print("")
        for src_state, dst_state in zip(src_indices, dst_indices):
            print(f"{src_state:0{nqubits}b} => {dst_state:0{nqubits-1}b}")
    res = density_matrix[np.ix_(src_indices, src_indices)]
    if normalize:
        res /= np.linalg.norm(res)
    return res


if __name__ == "__main__":
    nqubits = 5
    statevector = np.random.random_sample(2**nqubits)
    statevector /= np.linalg.norm(statevector)
    density_matrix = np.outer(statevector, statevector.conj())

    postselected_statevector = postselect_statevector(statevector, nqubits, 4, 0)
    postselected_density_matrix = postselect_density_matrix(
        density_matrix, nqubits, 4, 0
    )

    check_postselected_density_matrix = np.outer(
        postselected_statevector, postselected_statevector.conj()
    )

    print(np.abs(postselected_density_matrix - check_postselected_density_matrix).max())
