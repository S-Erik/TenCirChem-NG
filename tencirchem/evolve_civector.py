#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from functools import partial
import logging
from typing import Tuple

import numpy as np
from openfermion import jordan_wigner

from tencirchem import rdtypestr, uint_type
from tencirchem.misc import ex_op_to_fop
from tencirchem.hamiltonian import apply_op
from tencirchem.ci_utils import get_ci_strings, get_addr, get_init_civector


logger = logging.getLogger(__name__)


def get_fket_permutation(f_idx, n_qubits, n_elec_s, ci_strings, strs2addr):
    mask = 0
    for i in f_idx:
        mask += 1 << i
    excitation = ci_strings ^ mask
    return get_addr(excitation, n_qubits, n_elec_s, strs2addr)


def get_fket_phase(f_idx, ci_strings):
    if len(f_idx) == 2:
        mask1 = 1 << f_idx[0]
        mask2 = 1 << f_idx[1]
    else:
        assert len(f_idx) == 4
        mask1 = (1 << f_idx[0]) + (1 << f_idx[1])
        mask2 = (1 << f_idx[2]) + (1 << f_idx[3])
    flip = ci_strings ^ mask1
    mask = mask1 | mask2
    masked = flip & mask
    positive = masked == mask
    negative = masked == 0
    return positive, negative


FERMION_PHASE_MASK_CACHE = {}


def get_fermion_phase(f_idx, n_qubits, ci_strings):
    if f_idx in FERMION_PHASE_MASK_CACHE:
        mask, sign = FERMION_PHASE_MASK_CACHE[f_idx]
    else:
        # fermion operator index, not sorted
        fop = ex_op_to_fop(f_idx)

        # pauli string index, already sorted
        qop = jordan_wigner(fop)
        mask_str = ["0"] * n_qubits
        for idx, term in next(iter(qop.terms.keys())):
            if term != "Z":
                assert idx in f_idx
                continue
            mask_str[n_qubits - 1 - idx] = "1"
        mask = uint_type(int("".join(mask_str), base=2))

        if sorted(qop.terms.items())[0][1].real > 0:
            sign = -1
        else:
            sign = 1

        FERMION_PHASE_MASK_CACHE[f_idx] = mask, sign

    parity = ci_strings & mask
    assert parity.dtype in [np.uint32, np.uint64]
    if parity.dtype == np.uint32:
        mask = 0x11111111
        shift = 28
    else:
        mask = 0x1111111111111111
        shift = 60
    parity ^= parity >> 1
    parity ^= parity >> 2
    parity = (parity & mask) * mask
    parity = (parity >> shift) & 1

    return sign * np.sign(parity - 0.5).astype(np.int8)


def get_operators(n_qubits, n_elec_s, strs2addr, f_idx, ci_strings):
    if len(set(f_idx)) != len(f_idx):
        raise ValueError(f"Excitation {f_idx} not supported")
    xp = np
    fket_permutation = get_fket_permutation(f_idx, n_qubits, n_elec_s, ci_strings, strs2addr)
    fket_phase = xp.zeros(len(ci_strings))
    positive, negative = get_fket_phase(f_idx, ci_strings)
    fket_phase -= positive
    fket_phase += negative
    fket_phase *= get_fermion_phase(f_idx, n_qubits, ci_strings)
    f2ket_phase = xp.zeros(len(ci_strings))
    f2ket_phase -= positive
    f2ket_phase -= negative

    return fket_permutation, fket_phase, f2ket_phase


CI_OPERATOR_BATCH_CACHE = {}
CI_OPERATOR_CACHE = {}


def get_operator_tensors(n_qubits, n_elec_s, ex_ops):
    xp = np
    batch_key = (xp, rdtypestr, n_qubits, n_elec_s, ex_ops)
    if batch_key in CI_OPERATOR_BATCH_CACHE:
        return CI_OPERATOR_BATCH_CACHE[batch_key]

    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec_s, strs2addr=True)

    xp = np
    fket_permutation_tensor = xp.zeros((len(ex_ops), len(ci_strings)), dtype=uint_type)
    fket_phase_tensor = xp.zeros((len(ex_ops), len(ci_strings)), dtype=np.int8)
    f2ket_phase_tensor = xp.zeros((len(ex_ops), len(ci_strings)), dtype=np.int8)
    for i, f_idx in enumerate(ex_ops):
        if 64 < len(ex_ops):
            logger.info((i, f_idx))
        op_key = (xp, rdtypestr, n_qubits, n_elec_s, f_idx)
        if op_key in CI_OPERATOR_CACHE:
            fket_permutation, fket_phase, f2ket_phase = CI_OPERATOR_CACHE[op_key]
        else:
            fket_permutation, fket_phase, f2ket_phase = get_operators(n_qubits, n_elec_s, strs2addr, f_idx, ci_strings)
            CI_OPERATOR_CACHE[op_key] = fket_permutation, fket_phase, f2ket_phase
        fket_permutation_tensor[i] = fket_permutation
        fket_phase_tensor[i] = fket_phase
        f2ket_phase_tensor[i] = f2ket_phase

    fket_permutation_tensor = np.asarray(fket_permutation_tensor)
    fket_phase_tensor = np.asarray(fket_phase_tensor)
    f2ket_phase_tensor = np.asarray(f2ket_phase_tensor)

    ret = ci_strings, fket_permutation_tensor, fket_phase_tensor, f2ket_phase_tensor
    CI_OPERATOR_BATCH_CACHE[batch_key] = ret
    return ret


def get_theta_tensors(params, param_ids):
    theta_list = []
    for param_id in param_ids:
        theta_list.append(params[param_id])

    theta_tensor = np.asarray(theta_list)
    theta_sin_tensor = np.sin(theta_tensor)
    theta_1mcos_tensor = 1 - np.cos(theta_tensor)
    return theta_sin_tensor, theta_1mcos_tensor


def evolve_civector_by_tensor(
    civector, fket_permutation_tensor, fket_phase_tensor, f2ket_phase_tensor, theta_sin, theta_1mcos
):
    def _evolve_excitation(j, _civector):
        _fket_phase = fket_phase_tensor[j]
        _fket_permutation = fket_permutation_tensor[j]
        fket = _civector[_fket_permutation] * _fket_phase
        f2ket = f2ket_phase_tensor[j] * _civector
        _civector += theta_1mcos[j] * f2ket + theta_sin[j] * fket
        return _civector

    val = civector
    for i in range(0, len(fket_permutation_tensor)):
        val = _evolve_excitation(i, val)
    return val


def get_civector(params, n_qubits, n_elec_s, ex_ops, param_ids, init_state=None):
    ci_strings, fket_permutation_tensor, fket_phase_tensor, f2ket_phase_tensor = get_operator_tensors(
        n_qubits, n_elec_s, ex_ops
    )
    theta_sin, theta_1mcos = get_theta_tensors(params, param_ids)

    if init_state is None:
        civector = get_init_civector(len(ci_strings))
    else:
        civector = np.asarray(init_state)
    civector = evolve_civector_by_tensor(
        civector, fket_permutation_tensor, fket_phase_tensor, f2ket_phase_tensor, theta_sin, theta_1mcos
    )

    return civector.reshape(-1)


def evolve_excitation_nocache(civector, fket_permutation, fket_phase, f2ket_phase, theta_1mcos, theta_sin):
    fket = civector[fket_permutation] * fket_phase
    f2ket = civector * f2ket_phase
    civector += theta_1mcos * f2ket + theta_sin * fket
    return civector


def get_civector_nocache(params, n_qubits, n_elec_s, ex_ops, param_ids, init_state=None):
    theta_sin_tensor, theta_1mcos_tensor = get_theta_tensors(params, param_ids)
    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec_s, strs2addr=True)

    if init_state is None:
        civector = get_init_civector(len(ci_strings))
    else:
        civector = np.asarray(init_state)

    for theta_sin, theta_1mcos, f_idx in zip(theta_sin_tensor, theta_1mcos_tensor, ex_ops):
        fket_permutation, fket_phase, f2ket_phase = get_operators(n_qubits, n_elec_s, strs2addr, f_idx, ci_strings)
        civector = evolve_excitation_nocache(
            civector, fket_permutation, fket_phase, f2ket_phase, theta_1mcos, theta_sin
        )

    return civector.reshape(-1)


def get_energy_and_grad_civector_nocache(
    params, hamiltonian, n_qubits, n_elec_s, ex_ops: Tuple, param_ids: Tuple, init_state=None
):
    ket = get_civector_nocache(params, n_qubits, n_elec_s, ex_ops, param_ids, init_state)
    bra = apply_op(hamiltonian, ket)
    energy = bra @ ket

    gradients_beforesum = _get_gradients_civector_nocache(bra, ket, params, n_qubits, n_elec_s, ex_ops, param_ids)
    gradients_beforesum = np.asarray(gradients_beforesum)

    gradients = np.zeros(params.shape)
    for grad, param_id in zip(gradients_beforesum, param_ids):
        gradients[param_id] += grad

    return energy, 2 * gradients


def _get_gradients_civector_nocache(bra, ket, params, n_qubits, n_elec_s, ex_ops, param_ids):
    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec_s, True)
    theta_sin_tensor, theta_1mcos_tensor = get_theta_tensors(params, param_ids)

    gradients_beforesum = []
    for theta_sin, theta_1mcos, f_idx in reversed(list(zip(theta_sin_tensor, theta_1mcos_tensor, ex_ops))):
        fket_permutation, fket_phase, f2ket_phase = get_operators(n_qubits, n_elec_s, strs2addr, f_idx, ci_strings)
        bra = evolve_excitation_nocache(bra, fket_permutation, fket_phase, f2ket_phase, theta_1mcos, -theta_sin)
        ket = evolve_excitation_nocache(ket, fket_permutation, fket_phase, f2ket_phase, theta_1mcos, -theta_sin)
        fket = ket[fket_permutation] * fket_phase
        grad = bra @ fket
        gradients_beforesum.append(grad)
    gradients_beforesum = list(reversed(gradients_beforesum))
    gradients_beforesum = np.asarray(gradients_beforesum)

    return gradients_beforesum


def apply_excitation_civector(civector, n_qubits, n_elec_s, f_idx):
    _, fket_permutation, fket_phase, _ = get_operator_tensors(n_qubits, n_elec_s, ex_ops=(f_idx,))
    return civector[fket_permutation[0]] * fket_phase[0]


def apply_excitation_civector_nocache(civector, n_qubits, n_elec_s, f_idx):
    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec_s, strs2addr=True)
    fket_permutation, fket_phase, _ = get_operators(n_qubits, n_elec_s, strs2addr, f_idx, ci_strings)
    return civector[fket_permutation] * fket_phase
