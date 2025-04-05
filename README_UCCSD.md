# UCCSD energy workflow

We create a `UCCSD` object with `from_integral` and call `energy` with a parameter array `params`. We work with the `"civector"` engine.

### Broad overview
The first call to `energy` is different then subsequent calls. In the first call the `hamiltonian`/`fci_func` function is defined (and saved in a hashmap `hamiltonian_lib`) which takes a ci-vector, applies the hamiltonian to this ci-vector and returns the resulting ci-vector. `param_ids` is set to `range(len(ex_ops))` if it was `None`. 


### In-depth
1. `UCCSD.energy` -> `UCC.energy`
2. `UCC._check_params_argument`: if `params` are given then number of params are checked and then converted to a tensor with `tc.backend.convert_to_tensor(params).astype(tc.rdtypestr)`
3. Back to `UCC.energy`. Get hamiltonian in `_get_hamiltonian_and_core`. `hype="fcifunc"`. `hamiltonian_lib` starts as an empty hashmap. Get `hamiltonian` with `get_h_from_integral` -> `get_h_fcifunc_from_integral` since `hcb=False` (`hcb`: Whether force electrons to pair as hard-core boson (HCB). Default to False.)
4. `get_h_fcifunc_from_integral` uses `pyscf.fci.direct_nosym` to contract absorb one-electron part into two-electron part. `fci_func` function IS the `hamiltonian`. It gets a ci-vector and uses `direct_nosym.contract_2e` to apply the elec. struct. hamiltonian to that ci-vector and returns the resulting ci-vector (the energy is not returned, only the ci-vector). To support spin-dependent hamiltonians we might simply use a spin-dependent version of `direct_nosym`.
5. `fci_func` is returned to `get_h_from_integral` which returns `fci_func` (called `hamiltonian` now) itself to `_get_hamiltonian_and_core` (still called `hamiltonian`) which returns `fci_func, e_core, engine` (`fci_func` still called `hamiltonian`) to `energy`. In `_get_hamiltonian_and_core` the `hamiltonian`/`fci_func` function is saved in the `hamiltonian_lib` hashmap. So, subsequent calls to `_get_hamiltonian_and_core` access the `hamiltonian`/`fci_func` function via the `hamiltonian_lib`.
6. `hamiltonian` in the `energy` function is a function that takes a ci-vector, applies the hamiltonian to this ci-vector and returns the resulting ci-vector (not the energy)
7. `get_energy` gets called. In there
    ```python
        if param_ids is None:
            param_ids = range(len(ex_ops))
    ```
8. `get_ci_strings` gets called, `n_qubits` is two times the number of active orbitals (see `UCC.__init__`), `hcb` is still `False`. `xp` is either `numpy` or `cupy`, whichever is defined as the tencirchem backend. `cistring.make_strings` returns list of ci-vectors in binary format for all spatial orbitals (`n_qubits//2`) for given number of electrons. `cistring.make_strings` is called for both spin-channels.  Both spin-channels are then combined in `ci_strings = ((alpha << (n_qubits // 2)).reshape(-1, 1) + beta.reshape(1, -1)).ravel()` (TODO: investigate this line further!). `strs2addr` is False. The combined `ci_strings` are returned to `get_energy`
9. `translate_init_state` gets called and returns `None` since `init_state` is `None`. (`(1 << n_qubits)` is equal to `2**n_qubits`.)
10. `engine` is still equal to `"civector"`. `GETVECTOR_MAP[engine]` yields the function `get_civector_`
11. `get_civector_` (`tencirchem.static.evolve_civector.get_civector`) gets called and returns the ci-vector produced by the UCCSD ansatz. `init_state` equals `None`
12. `get_operator_tensors` gets called and returns all ci-strings and the "operator tensor" (a tuple of tensors). `get_ci_strings` is called as above but this time with `strs2addr=True` (TODO: What does returned `strs2addr` look like?). The "operator tensors" are saved in a hashmap for faster access. For the first time the "operator tensors" are calculated with `get_operators` and put in the hashmap. An "operator tensor" is a tuple of a `fket_permutation_tensor`, a `fket_phase_tensor` and a `f2ket_phase_tensor` where each of these individual tensors is a 2D array. We iterate over all excitations in `ex_ops` and for each excitation we calculate and set one row of `fket_permutation_tensor`, a `fket_phase_tensor` and a `f2ket_phase_tensor`.
13. `get_operators` gets called. `f_idx` is an element of the `ex_ops` list and therefore is an exication tuple, e.g. `(0, 1)` for a single excitation or `(6, 2, 0, 1)` for a double excitation. 
14. `get_fket_permutation` gets called.
15. `get_fket_phase` gets called.
16. `get_fermion_phase` gets called.
17. `fket_permutation, fket_phase, f2ket_phase` are returned to `get_civector` representing ...(TODO: math)
18. `get_theta_tensors` gets called calculating and returning ...(TODO: math)
19. `get_init_civector` gets called since `init_state` is `None` and returns ...
20. `eveolve_civector_by_tensor` gets called

25. `apply_op` calls the `hamiltonian`/`fci_func` function with `ket` to obtain the ci-vetor after applying the hamiltonian, saved in `hket`
26. Then simply the energy is returned by calculating `ket @ hket`

