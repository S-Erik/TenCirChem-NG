#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


import numpy as np
from pyscf.fci import cistring

from tencirchem import rdtypestr, uint_type
from tencirchem.misc import unpack_nelec


def get_ci_strings(n_qubits, n_elec_s, strs2addr=False):
    if 2**n_qubits > np.iinfo(uint_type).max:
        raise ValueError(f"Too many qubits: {n_qubits}, try using complex128 datatype")
    na, nb = unpack_nelec(n_elec_s)
    beta = cistring.make_strings(range(n_qubits // 2), nb)
    beta = np.array(beta, dtype=uint_type)
    if na == nb:
        alpha = beta
    else:
        alpha = cistring.make_strings(range(n_qubits // 2), na)
        alpha = np.array(alpha, dtype=uint_type)
    ci_strings = ((alpha << (n_qubits // 2)).reshape(-1, 1) + beta.reshape(1, -1)).ravel()
    if strs2addr:
        if na == nb:
            strs2addr = np.zeros(2 ** (n_qubits // 2), dtype=uint_type)
            strs2addr[beta] = np.arange(len(beta))
        else:
            strs2addr = np.zeros((2, 2 ** (n_qubits // 2)), dtype=uint_type)
            strs2addr[0][alpha] = np.arange(len(alpha))
            strs2addr[1][beta] = np.arange(len(beta))
        return ci_strings, strs2addr

    return ci_strings


def get_addr(excitation, n_qubits, n_elec_s, strs2addr, num_strings=None):
    # Right-shift bitstring by (n_qubits // 2) bits
    # yielding alpha spin part (left-side bits) of `excitation`
    alpha = excitation >> (n_qubits // 2)
    # 2**n is bitstring with a one at index n (counting from right starting from 0):
    #               0..0   1 0   ..0 0 0
    #   indices:    ...n+1 n n-1 ..2 1 0
    # 2**n - 1 is bitstring with a zero at index n and ones otherwise:
    #               0..0   0 1   ..1 1 1
    #   indices:    ...n+1 n n-1 ..2 1 0
    # Bitwise AND (&) now gives beta spin (right-side bits) of excitation
    beta = excitation & (2 ** (n_qubits // 2) - 1)
    na, nb = n_elec_s
    if na == nb:
        alpha_addr = strs2addr[alpha]
        beta_addr = strs2addr[beta]
    else:
        alpha_addr = strs2addr[0][alpha]
        beta_addr = strs2addr[1][beta]
    if num_strings is None:
        num_strings = cistring.num_strings(n_qubits // 2, nb)
    # print("In get_addr")
    # print(f"excitation: {[bin(x)[2:].zfill(n_qubits) for x in excitation]}")
    # print(f"alpha:      {[(bin(x)[2:].zfill(n_qubits), x.item()) for x in alpha]}")
    # print(f"beta:       {[(bin(x)[2:].zfill(n_qubits), x.item()) for x in beta]}")
    # print(f"strs2addr: {strs2addr}")
    # print(f"alpha_addr: {alpha_addr}, beta_addr: {beta_addr}")
    # print(f"num_strings: {num_strings}")
    # print(f"alpha_addr * num_strings + beta_addr: {alpha_addr * num_strings + beta_addr}")
    # print()
    # TODO: Unclear what strs2addr and return value represent
    return alpha_addr * num_strings + beta_addr


def get_ex_bitstring(n_qubits, n_elec_s, ex_op):
    na, nb = n_elec_s
    bitstring_basea = ["0"] * (n_qubits // 2 - na) + ["1"] * na
    bitstring_baseb = ["0"] * (n_qubits // 2 - nb) + ["1"] * nb
    bitstring_base = bitstring_basea + bitstring_baseb

    bitstring = bitstring_base.copy()[::-1]
    # first annihilation then creation
    if len(ex_op) == 2:
        bitstring[ex_op[1]] = "0"
        bitstring[ex_op[0]] = "1"
    else:
        assert len(ex_op) == 4
        bitstring[ex_op[3]] = "0"
        bitstring[ex_op[2]] = "0"
        bitstring[ex_op[1]] = "1"
        bitstring[ex_op[0]] = "1"

    return "".join(reversed(bitstring))


def civector_to_statevector(civector, n_qubits, ci_strings):
    statevector = np.zeros(2**n_qubits, dtype=rdtypestr)
    statevector[ci_strings] = civector
    return statevector


def statevector_to_civector(statevector, ci_strings):
    return statevector[ci_strings]


def get_init_civector(len_ci):
    civector = np.zeros(len_ci, dtype=rdtypestr)
    civector[0] = 1
    return civector
