#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.

from typing import Tuple

from openfermion import FermionOperator
from openfermion.utils import hermitian_conjugated


def format_ex_op(ex_op: Tuple) -> str:
    if len(ex_op) == 2:
        return f"{ex_op[0]}^ {ex_op[1]}"
    else:
        assert len(ex_op) == 4
        return f"{ex_op[0]}^ {ex_op[1]}^ {ex_op[2]} {ex_op[3]}"


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
