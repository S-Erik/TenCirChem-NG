#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


import logging
from inspect import isfunction

import numpy as np
from pyscf.fci import direct_nosym

from tencirchem import rdtypestr


logger = logging.getLogger(__name__)


def get_h_fcifunc_from_integral(int1e, int2e, n_elec):
    n_orb = len(int1e)
    h2e = direct_nosym.absorb_h1e(int1e, int2e, n_orb, n_elec, 0.5)

    def fci_func(civector):
        civector = np.asarray(civector).astype(np.float64)
        civector = direct_nosym.contract_2e(h2e, civector, norb=n_orb, nelec=n_elec)
        return np.asarray(civector).astype(rdtypestr)

    return fci_func


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
