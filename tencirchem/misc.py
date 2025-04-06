#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from functools import wraps
from inspect import isfunction
from typing import List, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from renormalizer import Model, Mpo, Op
from renormalizer.model.basis import BasisSet
from openfermion import FermionOperator, QubitOperator, jordan_wigner, get_sparse_operator
from openfermion.utils import hermitian_conjugated
from qiskit.quantum_info import SparsePauliOp
import tensorcircuit as tc

from tencirchem import rdtypestr
from tencirchem.constants import DISCARD_EPS


def csc_to_coo(csc):
    coo = coo_matrix(csc)
    mask = DISCARD_EPS < np.abs(coo.data.real)
    indices = np.array([coo.row[mask], coo.col[mask]]).T
    values = coo.data.real[mask].astype(rdtypestr)
    return tc.backend.coo_sparse_matrix(indices=indices, values=values, shape=coo.shape)


def fop_to_coo(fop: FermionOperator, n_qubits: int, real: bool = True):
    op = get_sparse_operator(jordan_wigner(reverse_fop_idx(fop, n_qubits)), n_qubits=n_qubits)
    if real:
        op = op.real
    return csc_to_coo(op)


def hcb_to_coo(qop: QubitOperator, n_qubits: int, real: bool = True):
    op = get_sparse_operator(qop, n_qubits)
    if real:
        op = op.real
    return csc_to_coo(op)


def qop_to_qiskit(qop: QubitOperator, n_qubits: int) -> SparsePauliOp:
    sparse_list = []
    for k, v in qop.terms.items():
        s = "".join(kk[1] for kk in k)
        idx = [kk[0] for kk in k]
        sparse_list.append([s, idx, v])
    return SparsePauliOp.from_sparse_list(sparse_list, num_qubits=n_qubits)


def reverse_qop_idx(op: QubitOperator, n_qubits: int) -> QubitOperator:
    ret = QubitOperator()
    for pauli_string, v in op.terms.items():
        # internally QubitOperator assumes ascending index
        pauli_string = tuple(reversed([(n_qubits - 1 - idx, symbol) for idx, symbol in pauli_string]))
        ret.terms[pauli_string] = v
    return ret


def reverse_fop_idx(op: FermionOperator, n_qubits: int) -> FermionOperator:
    ret = FermionOperator()
    for word, v in op.terms.items():
        word = tuple([(n_qubits - 1 - idx, symbol) for idx, symbol in word])
        ret.terms[word] = v
    return ret


def format_ex_op(ex_op: Tuple) -> str:
    if len(ex_op) == 2:
        return f"{ex_op[0]}^ {ex_op[1]}"
    else:
        assert len(ex_op) == 4
        return f"{ex_op[0]}^ {ex_op[1]}^ {ex_op[2]} {ex_op[3]}"


def canonical_mo_coeff(mo_coeff: np.ndarray):
    # make the first large element positive
    # all elements smaller than 1e-5 is highly unlikely (at least 1e10 basis)
    largest_elem_idx = np.argmax(1e-5 < np.abs(mo_coeff), axis=0)
    largest_elem = mo_coeff[(largest_elem_idx, np.arange(len(largest_elem_idx)))]
    return mo_coeff * np.sign(largest_elem).reshape(1, -1)


def get_n_qubits(vector_or_matrix_or_mpo_func):
    if isinstance(vector_or_matrix_or_mpo_func, list):
        return len(vector_or_matrix_or_mpo_func)
    if isfunction(vector_or_matrix_or_mpo_func):
        return vector_or_matrix_or_mpo_func.n_qubit
    # if hasattr(vector_or_matrix_or_mpo_func, "n_qubits"):
    #     return getattr(vector_or_matrix_or_mpo_func, "n_qubits")
    return round(np.log2(vector_or_matrix_or_mpo_func.shape[0]))


def ex_op_to_fop(ex_op, with_conjugation=False):
    if len(ex_op) == 2:
        fop = FermionOperator(f"{ex_op[0]}^ {ex_op[1]}")
    else:
        assert len(ex_op) == 4
        fop = FermionOperator(f"{ex_op[0]}^ {ex_op[1]}^ {ex_op[2]} {ex_op[3]}")
    if with_conjugation:
        fop = fop - hermitian_conjugated(fop)
    return fop


def unpack_nelec(n_elec_s):
    if isinstance(n_elec_s, tuple):
        na, nb = n_elec_s
    elif isinstance(n_elec_s, int):
        na, nb = n_elec_s // 2, n_elec_s // 2
    else:
        raise TypeError(f"Unknown electron number specification: {n_elec_s}")
    return na, nb
