from __future__ import annotations

import numpy as np

import tencirchem as tcc


def get_ex1_ops(norb, nelec) -> list[tuple[int, int]]:
    """Get one-body excitation operators.

    Parameters
    ----------
    norb: int
        Number of spatial orbitals
    nelec: tuple(int, int)
        Number of (alpha, beta) electrons

    Returns
    -------
    ex_op: List[Tuple]
        The excitation operators. Each operator is represented by a tuple of ints.

    """
    assert nelec[0] == nelec[1]
    # single excitations
    no = nelec[0]
    nv = norb - no

    ex1_ops = []
    for i in range(no):
        for a in range(nv):
            # alpha to alpha
            ex_op_a = (2 * no + nv + a, no + nv + i)
            # beta to beta
            ex_op_b = (no + a, i)
            ex1_ops.extend([ex_op_a, ex_op_b])
    return ex1_ops


def get_ex2_ops(norb: int, nelec: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """Get two-body excitation operators.

    Parameters
    ----------
    norb: int
        Number of spatial orbitals
    nelec: tuple(int, int)
        Number of (alpha, beta) electrons

    Returns
    -------
    ex_op: List[Tuple]
        The excitation operators. Each operator is represented by a tuple of ints.

    """
    assert nelec[0] == nelec[1]
    no = nelec[0]
    nv = norb - no

    def alpha_o(_i: int) -> int:
        return no + nv + _i

    def alpha_v(_i: int) -> int:
        return 2 * no + nv + _i

    def beta_o(_i: int) -> int:
        return _i

    def beta_v(_i: int) -> int:
        return no + _i

    # double excitations
    ex_ops = []
    # 2 alphas or 2 betas
    for i in range(no):
        for j in range(i):
            for a in range(nv):
                for b in range(a):
                    # i correspond to a and j correspond to b, as in PySCF convention
                    # otherwise the t2 amplitude has incorrect phase
                    # 2 alphas
                    ex_op_aa = (alpha_v(b), alpha_v(a), alpha_o(i), alpha_o(j))
                    # 2 betas
                    ex_op_bb = (beta_v(b), beta_v(a), beta_o(i), beta_o(j))
                    ex_ops.extend([ex_op_aa, ex_op_bb])
    assert len(ex_ops) == 2 * (no * (no - 1) / 2) * (nv * (nv - 1) / 2)
    # 1 alpha + 1 beta
    for i in range(no):
        for j in range(i + 1):
            for a in range(nv):
                for b in range(a + 1):
                    # i correspond to a and j correspond to b, as in PySCF convention
                    # otherwise the t2 amplitude has incorrect phase
                    if i == j and a == b:
                        # paired
                        ex_op_ab = (beta_v(a), alpha_v(a), alpha_o(i), beta_o(i))
                        ex_ops.append(ex_op_ab)
                        continue
                    # simple reflection
                    ex_op_ab1 = (beta_v(b), alpha_v(a), alpha_o(i), beta_o(j))
                    ex_op_ab2 = (alpha_v(b), beta_v(a), beta_o(i), alpha_o(j))
                    ex_ops.extend([ex_op_ab1, ex_op_ab2])
                    if (i != j) and (a != b):
                        # exchange alpha and beta
                        ex_op_ab3 = (beta_v(a), alpha_v(b), alpha_o(i), beta_o(j))
                        ex_op_ab4 = (alpha_v(a), beta_v(b), beta_o(i), alpha_o(j))
                        ex_ops.extend([ex_op_ab3, ex_op_ab4])

    return ex_ops
