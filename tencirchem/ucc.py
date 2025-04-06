#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from itertools import product
from collections import defaultdict
from time import time
import logging
from typing import Any, Tuple, List, Union

import numpy as np
from scipy.special import comb
import pandas as pd
from pyscf.cc.addons import spatial2spin
from pyscf.fci import direct_nosym

from tencirchem import rdtypestr
from tencirchem.constants import DISCARD_EPS
from tencirchem.engine_ucc import (
    get_civector,
    get_statevector,
    get_energy,
    apply_excitation,
    translate_init_state,
)
from tencirchem.ci_utils import get_ci_strings, get_ex_bitstring, get_addr, get_init_civector
from tencirchem.evolve_civector import get_civector_nocache
from tencirchem.evolve_civector import get_civector as get_civector_
from tencirchem.hamiltonian import apply_op

logger = logging.getLogger(__name__)


Tensor = Any


class UCC:
    """
    Base class for :class:`UCCSD`.
    """

    @classmethod
    def from_integral(
        cls,
        int1e: np.ndarray,
        int2e: np.ndarray,
        n_elec: Union[int, Tuple[int, int]],
        e_core: float = 0,
        ovlp: np.ndarray = None,
        **kwargs,
    ):
        """
        Construct UCC classes from electron integrals.

        Parameters
        ----------
        int1e: np.ndarray
            One-body integral in spatial orbital.
        int2e: np.ndarray
            Two-body integral, in spatial orbital, chemists' notation, and without considering symmetry.
        n_elec: int or tuple
            The number of electrons, or numbers of alpha/beta electrons
        e_core: float, optional
            The nuclear energy or core energy if active space approximation is involved.
            Defaults to 0.
        ovlp: np.ndarray
            The overlap integral. Defaults to ``None`` and identity matrix is used.
        kwargs:
            Other arguments to be passed to the :func:`__init__` function such as ``engine``.

        Returns
        -------
        ucc: :class:`UCC`
             A UCC instance
        """
        if isinstance(n_elec, tuple):
            spin = abs(n_elec[0] - n_elec[1])
            n_elec = n_elec[0] + n_elec[1]
        else:
            assert n_elec % 2 == 0
            spin = 0
        return cls(int1e, int2e, n_elec, spin, e_core, ovlp, **kwargs)

    def __init__(
        self,
        int1e,
        int2e,
        n_elec,
        spin,
        e_core,
        ovlp,
        init_method="mp2",
        active_space=None,
        aslst=None,
        engine=None,
        run_hf=True,
        run_mp2=True,
        run_ccsd=True,
        run_fci=True,
    ):
        r"""
        Initialize the class with molecular input.

        Parameters
        ----------
        int1e: np.ndarray
            One-body integral in spatial orbital.
        int2e: np.ndarray
            Two-body integral, in spatial orbital, chemists' notation, and without considering symmetry.
        n_elec: int or tuple
            The number of electrons, or numbers of alpha/beta electrons
        e_core: float, optional
            The nuclear energy or core energy if active space approximation is involved.
            Defaults to 0.
        ovlp: np.ndarray
            The overlap integral. Defaults to ``None`` and identity matrix is used.
        init_method: str, optional
            How to determine the initial amplitude guess. Accepts ``"mp2"`` (default), ``"ccsd"``, ``"fe"``
            and ``"zeros"``.
        active_space: Tuple[int, int], optional
            Active space approximation. The first integer is the number of electrons and the second integer is
            the number or spatial-orbitals. Defaults to None.
        aslst: List[int], optional
            Pick orbitals for the active space. Defaults to None which means the orbitals are sorted by energy.
            The orbital index is 0-based.

            .. note::
                See `PySCF document <https://pyscf.org/user/mcscf.html#picking-an-active-space>`_
                for choosing the active space orbitals. Here orbital index is 0-based, whereas in PySCF by default it
                is 1-based.

        engine: str, optional
            The engine to run the calculation. See :ref:`advanced:Engines` for details.
        run_hf: bool, optional
            Whether run HF for molecule orbitals. Defaults to ``True``.
            The argument has no effect if ``mol`` is a ``RHF`` object.
        run_mp2: bool, optional
            Whether run MP2 for initial guess and energy reference. Defaults to ``True``.
        run_ccsd: bool, optional
            Whether run CCSD for initial guess and energy reference. Defaults to ``True``.
        run_fci: bool, optional
            Whether run FCI  for energy reference. Defaults to ``True``.

        See Also
        --------
        tencirchem.UCCSD
        tencirchem.KUPCCGSD
        tencirchem.PUCCD
        """
        self.int1e = int1e
        self.int2e = int2e
        self.n_elec = n_elec
        self.spin = spin
        self.e_core = e_core
        self.ovlp = ovlp

        nelectron = n_elec
        norb = self.int1e.shape[0]

        if active_space is None:
            active_space = (nelectron, int(norb))

        self.spin = spin
        self.n_qubits = 2 * active_space[1]

        # process activate space
        self.active_space = active_space
        self.n_elec = active_space[0]
        self.active = active_space[1]
        self.inactive_occ = (nelectron - active_space[0]) // 2
        assert (nelectron - active_space[0]) % 2 == 0
        self.inactive_vir = norb - active_space[1] - self.inactive_occ
        if aslst is None:
            aslst = list(range(self.inactive_occ, norb - self.inactive_vir))
        if len(aslst) != active_space[1]:
            raise ValueError("sort_mo should have the same length as the number of active orbitals.")
        self.aslst = aslst

        # process backend
        self._check_engine(engine)

        if engine is None:
            # no need to be too precise
            if self.n_qubits <= 16:
                engine = "civector"
            else:
                engine = "civector-large"
        self.engine = engine

        self.e_core = e_core

        # initial guess
        self.t1 = np.zeros([self.no, self.nv])
        self.t2 = np.zeros([self.no, self.no, self.nv, self.nv])
        self.init_method = init_method
        if init_method is None or init_method in ["zeros", "zero"]:
            pass
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

        # circuit related
        self._init_state = None
        self.ex_ops = []
        self._param_ids = None
        self.init_guess = None
        # for manually set
        self._params = None

    def _check_params_argument(self, params, strict=True):
        if params is None:
            if self.params is not None:
                params = self.params
            else:
                if strict:
                    raise ValueError("Run the `.kernel` method to determine the parameters first")
                else:
                    if self.init_guess is not None:
                        params = self.init_guess
                    else:
                        params = np.zeros(self.n_params)

        if len(params) != self.n_params:
            raise ValueError(f"Incompatible parameter shape. {self.n_params} is desired. Got {len(params)}")
        return np.asarray(params).astype(rdtypestr)

    def _check_engine(self, engine):
        supported_engine = ["civector"]
        if not engine in supported_engine:
            raise ValueError(f"Engine '{engine}' not supported")

    def _sanity_check(self):
        if self.ex_ops is None:
            raise ValueError("`ex_ops` or `param_ids` not defined")

    def civector(self, params: Tensor = None, engine: str | None = None) -> Tensor:
        """
        Evaluate the configuration interaction (CI) vector.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.

        Returns
        -------
        civector: Tensor
            Corresponding CI vector

        See Also
        --------
        statevector: Evaluate the circuit state vector.
        energy: Evaluate the total energy.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> uccsd.civector([0, 0])  # HF state
        array([1., 0., 0., 0.])
        """
        self._sanity_check()
        params = self._check_params_argument(params)
        self._check_engine(engine)
        if engine is None:
            engine = self.engine
        civector = get_civector(
            params, self.n_qubits, self.n_elec_s, self.ex_ops, self.param_ids, self.init_state, engine
        )
        return civector

    def get_ci_strings(self, strs2addr: bool = False) -> np.ndarray:
        """
        Get the CI bitstrings for all configurations in the CI vector.

        Parameters
        ----------
        strs2addr: bool, optional.
            Whether return the reversed mapping for one spin sector. Defaults to ``False``.

        Returns
        -------
        cistrings: np.ndarray
            The CI bitstrings.
        string_addr: np.ndarray
            The address of the string in one spin sector. Returned when ``strs2addr`` is set to ``True``.

        Examples
        --------
        >>> from tencirchem import UCCSD, PUCCD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> uccsd.get_ci_strings()
        array([ 5,  6,  9, 10], dtype=uint64)
        >>> [f"{bin(i)[2:]:0>4}" for i in uccsd.get_ci_strings()]
        ['0101', '0110', '1001', '1010']
        >>> uccsd.get_ci_strings(True)[1]  # only one spin sector
        array([0, 0, 1, 0], dtype=uint64)
        """
        return get_ci_strings(self.n_qubits, self.n_elec_s, strs2addr=strs2addr)

    # since there's ci_vector method
    ci_strings = get_ci_strings

    def get_addr(self, bitstring: str) -> int:
        """
        Get the address (index) of a CI bitstring in the CI vector.

        Parameters
        ----------
        bitstring: str
            The bitstring such as ``"0101"``.

        Returns
        -------
        address: int
            The bitstring address.

        Examples
        --------
        >>> from tencirchem import UCCSD, PUCCD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> uccsd.get_addr("0101")  # the HF state
        0
        >>> uccsd.get_addr("1010")
        3
        >>> puccd = PUCCD(h2)
        >>> puccd.get_addr("01")  # the HF state
        0
        >>> puccd.get_addr("10")
        1
        """
        _, strs2addr = self.get_ci_strings(strs2addr=True)
        return int(get_addr(int(bitstring, base=2), self.n_qubits, self.n_elec_s, strs2addr))

    def statevector(self, params: Tensor = None, engine: str = None) -> Tensor:
        """
        Evaluate the circuit state vector.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.

        Returns
        -------
        statevector: Tensor
            Corresponding state vector

        See Also
        --------
        civector: Evaluate the configuration interaction (CI) vector.
        energy: Evaluate the total energy.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> uccsd.statevector([0, 0])  # HF state
        array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        """
        self._sanity_check()
        params = self._check_params_argument(params)
        self._check_engine(engine)
        if engine is None:
            engine = self.engine
        statevector = get_statevector(
            params, self.n_qubits, self.n_elec_s, self.ex_ops, self.param_ids, self.init_state, engine
        )
        return statevector

    def energy(self, params: Tensor = None, engine: str | None = None) -> float:
        """
        Evaluate the total energy.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.

        Returns
        -------
        energy: float
            Total energy

        See Also
        --------
        civector: Get the configuration interaction (CI) vector.
        statevector: Evaluate the circuit state vector.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> round(uccsd.energy([0, 0]), 8)  # HF state
        -1.11670614
        """
        if engine is None:
            engine = self.engine
        self._sanity_check()
        params = self._check_params_argument(params)
        n_orb = len(self.int1e)

        h2e = direct_nosym.absorb_h1e(self.int1e, self.int2e, n_orb, self.n_elec_s, 0.5)  # type: ignore

        def fci_func(civector):
            civector = np.asarray(civector).astype(np.float64)
            civector = direct_nosym.contract_2e(h2e, civector, norb=n_orb, nelec=self.n_elec_s)
            return np.asarray(civector).astype(rdtypestr)

        ci_strings = get_ci_strings(self.n_qubits, self.n_elec_s)
        init_state = translate_init_state(self.init_state, self.n_qubits, ci_strings)
        ket = get_civector_(params, self.n_qubits, self.n_elec_s, tuple(self.ex_ops), init_state=init_state)
        hket = fci_func(ket)
        e = ket @ hket

        return float(e) + self.e_core

    def apply_excitation(self, state: Tensor, ex_op: Tuple, engine: str | None = None) -> Tensor:
        """
        Apply a given excitation operator to a given state.

        Parameters
        ----------
        state: Tensor
            The input state in statevector or CI vector form.
        ex_op: tuple of ints
            The excitation operator.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.
        Returns
        -------
        tensor: Tensor
            The resulting tensor.

        Examples
        --------
        >>> from tencirchem import UCC
        >>> from tencirchem.molecule import h2
        >>> ucc = UCC(h2)
        >>> ucc.apply_excitation([1, 0, 0, 0], (3, 1, 0, 2))
        array([0, 0, 0, 1])
        """
        self._check_engine(engine)
        if engine is None:
            engine = self.engine
        return apply_excitation(state, self.n_qubits, self.n_elec_s, ex_op, engine=engine)

    def _statevector_to_civector(self, statevector=None):
        if statevector is None:
            civector = self.civector()
        else:
            if len(statevector) == self.statevector_size:
                ci_strings = self.get_ci_strings()
                civector = statevector[ci_strings]
            else:
                if len(statevector) == self.civector_size:
                    civector = statevector
                else:
                    raise ValueError(f"Incompatible statevector size: {len(statevector)}")

        civector = np.asarray(civector)
        return civector

    @property
    def e_ucc(self) -> float:
        """
        Returns UCC energy
        """
        return self.energy()

    def print_ansatz(self):
        df_dict = {
            "#qubits": [self.n_qubits],
            "#params": [self.n_params],
            "#excitations": [len(self.ex_ops)],
        }
        if self.init_state is None:
            df_dict["initial condition"] = "RHF"
        else:
            df_dict["initial condition"] = "custom"
        print(pd.DataFrame(df_dict).to_string(index=False))

    def get_init_state_dataframe(self, coeff_epsilon: float = DISCARD_EPS) -> pd.DataFrame:
        """
        Returns initial state information dataframe.

        Parameters
        ----------
        coeff_epsilon: float, optional
            The threshold to screen out states with small coefficients.
            Defaults to 1e-12.

        Returns
        -------
        pd.DataFrame

        See Also
        --------
        init_state: The circuit initial state before applying the excitation operators.

        Examples
        --------
        >>> from tencirchem import UCC
        >>> from tencirchem.molecule import h2
        >>> ucc = UCC(h2)
        >>> ucc.init_state = [0.707, 0, 0, 0.707]
        >>> ucc.get_init_state_dataframe()   # doctest: +NORMALIZE_WHITESPACE
             configuration  coefficient
        0          0101        0.707
        1          1010        0.707
        """
        columns = ["configuration", "coefficient"]
        if self.init_state is None:
            init_state = get_init_civector(self.civector_size)
        else:
            init_state = self.init_state
        ci_strings = self.get_ci_strings()
        ci_coeffs = translate_init_state(init_state, self.n_qubits, ci_strings)
        data_list = []
        for ci_string, coeff in zip(ci_strings, ci_coeffs):
            if np.abs(coeff) < coeff_epsilon:
                continue
            ci_string = bin(ci_string)[2:]
            ci_string = "0" * (self.n_qubits - len(ci_string)) + ci_string
            data_list.append((ci_string, coeff))
        return pd.DataFrame(data_list, columns=columns)

    @property
    def n_elec_s(self):
        """The number of electrons for alpha and beta spin"""
        return (self.n_elec + self.spin) // 2, (self.n_elec - self.spin) // 2

    @property
    def no(self) -> int:
        """The number of occupied orbitals."""
        return self.n_elec // 2

    @property
    def nv(self) -> int:
        """The number of virtual (unoccupied orbitals)."""
        return self.active - self.no

    @property
    def n_params(self) -> int:
        """The number of parameter in the ansatz/circuit."""
        return len(self.ex_ops)

    @property
    def statevector_size(self) -> int:
        """The size of the statevector."""
        return 1 << self.n_qubits

    @property
    def civector_size(self) -> int:
        """
        The size of the CI vector.
        """
        na, nb = self.n_elec_s
        return round(comb(self.n_qubits // 2, na)) * round(comb(self.n_qubits // 2, nb))

    @property
    def init_state(self) -> Tensor:
        """
        The circuit initial state before applying the excitation operators. Usually RHF.

        See Also
        --------
        get_init_state_dataframe: Returns initial state information dataframe.
        """
        return self._init_state

    @init_state.setter
    def init_state(self, init_state):
        self._init_state = init_state

    @property
    def params(self) -> Tensor:
        """The circuit parameters."""
        if self._params is not None:
            return self._params
        return None

    @params.setter
    def params(self, params):
        self._params = params


def compute_fe_t2(no, nv, int1e, int2e):
    n_orb = no + nv

    def translate_o(n):
        if n % 2 == 0:
            return n // 2 + n_orb
        else:
            return n // 2

    def translate_v(n):
        if n % 2 == 0:
            return n // 2 + no + n_orb
        else:
            return n // 2 + no

    t2 = np.zeros((2 * no, 2 * no, 2 * nv, 2 * nv))
    for i, j, k, l in product(range(2 * no), range(2 * no), range(2 * nv), range(2 * nv)):
        # spin not conserved
        if i % 2 != k % 2 or j % 2 != l % 2:
            continue
        a = translate_o(i)
        b = translate_o(j)
        s = translate_v(l)
        r = translate_v(k)
        if len(set([a, b, s, r])) != 4:
            continue
        # r^ s^ b a
        rr, ss, bb, aa = [i % n_orb for i in [r, s, b, a]]
        if (r < n_orb and s < n_orb) or (r >= n_orb and s >= n_orb):
            e_inter = int2e[aa, rr, bb, ss] - int2e[aa, ss, bb, rr]
        else:
            e_inter = int2e[aa, rr, bb, ss]
        if np.allclose(e_inter, 0):
            continue
        e_diff = _compute_e_diff(r, s, b, a, int1e, int2e, n_orb, no)
        if np.allclose(e_diff, 0):
            raise RuntimeError("RHF degenerate ground state")
        theta = np.arctan(-2 * e_inter / e_diff) / 2
        t2[i, j, k, l] = theta
    return t2


def _compute_e_diff(r, s, b, a, int1e, int2e, n_orb, no):
    inert_a = list(range(no))
    inert_b = list(range(no))
    old_a = []
    old_b = []
    for i in [b, a]:
        if i < n_orb:
            inert_b.remove(i)
            old_b.append(i)
        else:
            inert_a.remove(i % n_orb)
            old_a.append(i % n_orb)

    new_a = []
    new_b = []
    for i in [r, s]:
        if i < n_orb:
            new_b.append(i)
        else:
            new_a.append(i % n_orb)

    diag1e = np.diag(int1e)
    diagj = np.einsum("iijj->ij", int2e)
    diagk = np.einsum("ijji->ij", int2e)

    e_diff_1e = diag1e[new_a].sum() + diag1e[new_b].sum() - diag1e[old_a].sum() - diag1e[old_b].sum()
    # fmt: off
    e_diff_j = _compute_j_outer(diagj, inert_a, inert_b, new_a, new_b) \
               - _compute_j_outer(diagj, inert_a, inert_b, old_a, old_b)
    e_diff_k = _compute_k_outer(diagk, inert_a, inert_b, new_a, new_b) \
               - _compute_k_outer(diagk, inert_a, inert_b, old_a, old_b)
    # fmt: on
    return e_diff_1e + 1 / 2 * (e_diff_j - e_diff_k)


def _compute_j_outer(diagj, inert_a, inert_b, outer_a, outer_b):
    # fmt: off
    v = diagj[inert_a][:, outer_a].sum() + diagj[outer_a][:, inert_a].sum() + diagj[outer_a][:, outer_a].sum() \
      + diagj[inert_a][:, outer_b].sum() + diagj[outer_a][:, inert_b].sum() + diagj[outer_a][:, outer_b].sum() \
      + diagj[inert_b][:, outer_a].sum() + diagj[outer_b][:, inert_a].sum() + diagj[outer_b][:, outer_a].sum() \
      + diagj[inert_b][:, outer_b].sum() + diagj[outer_b][:, inert_b].sum() + diagj[outer_b][:, outer_b].sum()
    # fmt: on
    return v


def _compute_k_outer(diagk, inert_a, inert_b, outer_a, outer_b):
    # fmt: off
    v = diagk[inert_a][:, outer_a].sum() + diagk[outer_a][:, inert_a].sum() + diagk[outer_a][:, outer_a].sum() \
      + diagk[inert_b][:, outer_b].sum() + diagk[outer_b][:, inert_b].sum() + diagk[outer_b][:, outer_b].sum()
    # fmt: on
    return v
