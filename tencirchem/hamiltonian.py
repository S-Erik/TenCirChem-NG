#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


import logging
from inspect import isfunction
from itertools import product
from typing import Tuple, List

import numpy as np
import tensornetwork as tn
from renormalizer import Mpo
from openfermion import FermionOperator, QubitOperator
from pyscf.scf.hf import RHF
from pyscf.mcscf import CASCI
from pyscf.fci import direct_nosym, cistring
from pyscf import ao2mo
from tensorcircuit import QuOperator

from tencirchem import rdtypestr
from tencirchem.constants import DISCARD_EPS


logger = logging.getLogger(__name__)


def get_h_fcifunc_from_integral(int1e, int2e, n_elec):
    n_orb = len(int1e)
    h2e = direct_nosym.absorb_h1e(int1e, int2e, n_orb, n_elec, 0.5)

    def fci_func(civector):
        civector = np.asarray(civector).astype(np.float64)
        civector = direct_nosym.contract_2e(h2e, civector, norb=n_orb, nelec=n_elec)
        return np.asarray(civector).astype(rdtypestr)

    return fci_func


def get_h_from_integral(int1e, int2e, n_elec_s, htype: str):
    if htype == "sparse":
        raise ValueError("Sparse hamiltonians not supported!")
    else:
        assert htype.lower() == "fcifunc"
        hamiltonian = get_h_fcifunc_from_integral(int1e, int2e, n_elec_s)
    return hamiltonian


def mpo_to_quoperator(mpo: Mpo):
    array_list = [m.array for m in mpo]
    # squeeze dim 1 out
    assert len(array_list) >= 2
    a, b, c, d = array_list[0].shape
    array_list[0] = array_list[0].reshape(b, c, d)
    a, b, c, d = array_list[-1].shape
    array_list[-1] = array_list[-1].reshape(a, b, c)
    # convert MPO to tensor-network
    node_list = [tn.Node(array) for array in array_list]

    node_list[0][2] ^ node_list[1][0]
    for i in range(1, len(node_list) - 1):
        node_list[i][3] ^ node_list[i + 1][0]

    in_edges = [node_list[0][0]] + [node[1] for node in node_list[1:]]
    out_edges = [node_list[0][1]] + [node[2] for node in node_list[1:]]

    qop = QuOperator(out_edges, in_edges)
    return qop


def apply_op(op, state):
    if isfunction(op):
        return op(state)
    else:
        return op @ state


def random_integral(nao: int, seed: int = 2077):
    np.random.seed(seed)
    int1e = np.random.uniform(-1, 1, size=(nao, nao))
    int2e = np.random.uniform(-1, 1, size=(nao, nao, nao, nao))
    int1e = 0.5 * (int1e + int1e.T)
    int2e = symmetrize_int2e(int2e)
    return int1e, int2e


def symmetrize_int2e(int2e):
    int2e = 0.25 * (
        int2e + int2e.transpose((0, 1, 3, 2)) + int2e.transpose((1, 0, 2, 3)) + int2e.transpose((2, 3, 0, 1))
    )
    int2e = 0.5 * (int2e + int2e.transpose(3, 2, 1, 0))
    return int2e
