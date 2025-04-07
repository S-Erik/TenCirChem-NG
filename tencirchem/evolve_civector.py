#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


import logging

import numpy as np
from openfermion import jordan_wigner

from tencirchem import rdtypestr, uint_type
from tencirchem.misc import ex_op_to_fop
from tencirchem.ci_utils import get_ci_strings, get_addr, get_init_civector


logger = logging.getLogger(__name__)


def get_fket_permutation(f_idx, n_qubits, n_elec_s, ci_strings, strs2addr):
    mask = 0  # Think of a bitstring 0...0000
    for i in f_idx:
        mask += 1 << i  # Adding 0..010..0 where 1 is at the i-th position from the right
        # mask += 2**i  # same as above
    # mask now is bitstring with a one at each index in f_idx list and zeros otherwise,
    # where f_idx is an excitation, e.g., (3, 0) or (6, 3, 1, 2)
    excitation = ci_strings ^ mask  # Bitwise XOR: Flips bits of bitstring ci_strings where mask has ones
    # print("In get_fket_permutation")
    # print(f"ex_op: {f_idx} -> mask: {bin(mask)[2:].zfill(n_qubits)}")
    # print(f"org ci-strings: {[bin(x)[2:].zfill(n_qubits) for x in ci_strings]}")
    # print(f"masked:         {[bin(x)[2:].zfill(n_qubits) for x in excitation]}")
    # print(f"strs2addr:      {strs2addr}\ttype: {type(strs2addr)}, type element: {type(strs2addr[0])}")
    # print()
    # TODO: Unclear what get_addr does here and what return value represents
    return get_addr(excitation, n_qubits, n_elec_s, strs2addr)


def get_fket_phase(f_idx, ci_strings):
    if len(f_idx) == 2:
        mask1 = 1 << f_idx[0]
        mask2 = 1 << f_idx[1]
    else:
        assert len(f_idx) == 4
        mask1 = (1 << f_idx[0]) + (1 << f_idx[1])  # creation operator indices
        mask2 = (1 << f_idx[2]) + (1 << f_idx[3])  # annihilation operator indices
    flip = ci_strings ^ mask1  # Flip bits for creation operator indices
    mask = mask1 | mask2  # Combine masks: ones at each index in f_idx and zeros otherwise
    # Bitwise AND (&). Produces a one where creation operators act on unoccpied orbitals
    #                                 and orbitals where annihilation operators act are already occupied
    masked = flip & mask
    # True/1 where `masked` matches exaclty `mask`
    # equivalent to: True/1 where the ci-string is not mapped to the zero state by the excitation
    # equivalent to: True/1 where the ci-string is not destroyed by the excitation
    # equivalent to: True/1 where the excitation a^† a (or a^† a^† a a) can act on the ci-string without destryoing the state
    # equivalent to: True/1 where the ci-string is UNOCCUPIED on EVERY orbital the creation operators act on
    #                       and OCCUPIED on EVERY orbital the annihilation operators act on
    positive = masked == mask
    # True/1 where `masked` is equal to all zero bitstring
    # equivalent to: True/1 where the excitation a a^† (or a a a^† a^†), which is the hermiatin conjugate of a^† a (or a^† a^† a a) from above,
    #                can act on the ci-string without destryoing the state
    # equivalent to: True/1 where the ci-string is OCCUPIED on EVERY orbital the creation operators act on
    #                       and UNOCCUPIED on EVERY orbital the annihilation operators act on
    negative = masked == 0
    # print("In get_fket_phase")
    # print(f"ex_op: {f_idx}")
    # print(f"ci_strings: {[bin(x)[2:] for x in ci_strings]}")
    # print(f"mask1:      {bin(mask1)[2:]}")
    # print(f"mask2:      {bin(mask2)[2:]}")
    # print(f"mask:       {bin(mask)[2:]}")
    # print(f"flip:       {[bin(x)[2:] for x in flip]}")
    # print(f"masked:     {[bin(x)[2:] for x in masked]}")
    # print(f"positive:   {[int(x) for x in positive]}")
    # print(f"negative:   {[int(x) for x in negative]}")
    # print()
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
        # mask is bitstring with one where Z operators act one when the excitation is mapped with the Jordan-Wigner mapper
        # TODO: This can be done more efficient, right? We do not need to do the JW mapping to know where the Z operators act on

        # TODO: What does the following mean physically?
        # TODO: For 10e 10o there is no sign=1. When is sign set to 1? For spin-polarized systems?
        # TODO: qop.terms.items() can be e.g.:
        #       dict_items([
        #           (((0, 'Y'), (1, 'Z'), (2, 'X')), 0.25j),
        #           (((0, 'X'), (1, 'Z'), (2, 'X')), (0.25+0j)),
        #           (((0, 'Y'), (1, 'Z'), (2, 'Y')), (0.25+0j)),
        #           (((0, 'X'), (1, 'Z'), (2, 'Y')), -0.25j)
        #       ])
        #       The dict_items get sorted by key and tuples are sorted element-by-element.
        #       [0][1] gives first value of (key, value) item of sorted items list
        #       (0, ...) < (1, ...) and (..., "X") < (..., "Y").
        #       Here we would get
        #           (((0, 'X'), (1, 'Z'), (2, 'X')), (0.25+0j)),
        #           ...
        #       We, therefore, always get keys only containing X's and Z's.
        #       There cannot be Y's since they would be in the place of the X's,
        if sorted(qop.terms.items())[0][1].real > 0:
            sign = -1
        else:
            sign = 1

        # print("In get_fermion_phase")
        # print(f"ex_op:                 {f_idx}")
        # print(f"fop:                   {fop}")
        # print(f"qop:                   {qop}")
        # print(f"qop.terms:             {qop.terms}")
        # print(f"qop.terms.items:       {qop.terms.items()}")
        # print(f"sorted qop:            {sorted(qop.terms.items())}")
        # print(f"sign:                  {sign}")
        # print(f"ci_strings:            {[bin(x)[2:].zfill(n_qubits) for x in ci_strings]}")
        # print(f"mask:                  {bin(mask)[2:].zfill(n_qubits)} = {mask}")
        # print()

        FERMION_PHASE_MASK_CACHE[f_idx] = mask, sign

    parity = ci_strings & mask
    print(f"parity:                {[bin(x)[2:].zfill(n_qubits) for x in parity]}")
    parity = np.asarray([bin(x)[2:].count("1") % 2 for x in parity])  # Check parity of bitstring, same as below, right?
    # Following checks if parity has an even/odd number of ones (then parity will be 0/1 before return)
    # assert parity.dtype in [np.uint32, np.uint64]
    # if parity.dtype == np.uint32:
    #     mask = 0x11111111
    #     shift = 28
    # else:
    #     mask = 0x1111111111111111
    #     shift = 60
    # parity ^= parity >> 1
    # parity ^= parity >> 2
    # parity = (parity & mask) * mask
    # parity = (parity >> shift) & 1

    return sign * np.sign(parity - 0.5).astype(np.int8)


def get_operators(n_qubits, n_elec_s, strs2addr, f_idx, ci_strings):
    if len(set(f_idx)) != len(f_idx):
        raise ValueError(f"Excitation {f_idx} not supported")
    # TODO: Unclear what the following does
    fket_permutation = get_fket_permutation(f_idx, n_qubits, n_elec_s, ci_strings, strs2addr)
    fket_phase = np.zeros(len(ci_strings))
    # positive and negative are lists. length is number of ci-strings
    # Each element in positive/negative says if the corresponding ci-string satisfies a certain condition.
    # The element is (0)1 if condition is (not) satisfied.
    # positive condition: excitation f_idx (a^† a OR a^† a^† a a) can act on the ci-string without destryoing the state
    # positive condition: excitation f_idx (a a^† OR a a a^† a^†) can act on the ci-string without destryoing the state
    # Remember an excitation in the UCC is e^(a^† a - a a^†) OR e^(a^† a^† a a - a a a^† a^†)
    positive, negative = get_fket_phase(f_idx, ci_strings)
    # TODO: -= since excitation generates |psi_0> - |psi_1>, where psi_0(1) is the ci-string before (after) applying the excitation?
    fket_phase -= positive
    fket_phase += negative
    fket_phase *= get_fermion_phase(f_idx, n_qubits, ci_strings)

    f2ket_phase = np.zeros(len(ci_strings))
    # TODO: Why -= for both positive and negative. Why not += positive and -=negative?
    f2ket_phase -= positive
    f2ket_phase -= negative

    return fket_permutation, fket_phase, f2ket_phase


CI_OPERATOR_BATCH_CACHE = {}
CI_OPERATOR_CACHE = {}


def get_operator_tensors(n_qubits, n_elec_s, ex_ops):
    batch_key = (rdtypestr, n_qubits, n_elec_s, ex_ops)
    if batch_key in CI_OPERATOR_BATCH_CACHE:
        return CI_OPERATOR_BATCH_CACHE[batch_key]

    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec_s, strs2addr=True)

    fket_permutation_tensor = np.zeros((len(ex_ops), len(ci_strings)), dtype=uint_type)
    fket_phase_tensor = np.zeros((len(ex_ops), len(ci_strings)), dtype=np.int8)
    f2ket_phase_tensor = np.zeros((len(ex_ops), len(ci_strings)), dtype=np.int8)
    for i, f_idx in enumerate(ex_ops):
        op_key = (rdtypestr, n_qubits, n_elec_s, f_idx)
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


def get_civector(params, n_qubits, n_elec_s, ex_ops, init_state=None):
    ci_strings, fket_permutation_tensor, fket_phase_tensor, f2ket_phase_tensor = get_operator_tensors(
        n_qubits, n_elec_s, ex_ops
    )
    theta_tensor = np.asarray(params)
    theta_sin = np.sin(theta_tensor)
    theta_1mcos = 1 - np.cos(theta_tensor)

    if init_state is None:
        civector = np.zeros(len(ci_strings), dtype=rdtypestr)
        civector[0] = 1
        # civector = get_init_civector(len(ci_strings)) # Same as above
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


def apply_excitation_civector(civector, n_qubits, n_elec_s, f_idx):
    _, fket_permutation, fket_phase, _ = get_operator_tensors(n_qubits, n_elec_s, ex_ops=(f_idx,))
    return civector[fket_permutation[0]] * fket_phase[0]


def apply_excitation_civector_nocache(civector, n_qubits, n_elec_s, f_idx):
    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec_s, strs2addr=True)
    fket_permutation, fket_phase, _ = get_operators(n_qubits, n_elec_s, strs2addr, f_idx, ci_strings)
    return civector[fket_permutation] * fket_phase
