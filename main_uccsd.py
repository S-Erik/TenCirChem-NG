import numpy as np

import tencirchem as tcc
from ex_ops import get_ex1_ops, get_ex2_ops

nelec = (1, 1)
norb = 2
h1e, h2e = tcc.hamiltonian.random_integral(norb)
ecore = 0.0

kwargs = {
    "init_method": "zeros",
    "engine": "civector",
    "run_hf": False,
    "run_mp2": False,
    "run_ccsd": False,
    "run_fci": False,
}
tcc_uccsd = tcc.ucc_all_in_one.UCC.from_integral(h1e, h2e, nelec, ecore, **kwargs)
tcc_uccsd.param_ids = None

single_ex = get_ex1_ops(norb, nelec)
double_ex = get_ex2_ops(norb, nelec)
tcc_uccsd.ex_ops = double_ex + single_ex

print(tcc_uccsd.ex_ops)
print(tcc_uccsd.param_ids)
print(tcc_uccsd.param_to_ex_ops)
print(tcc_uccsd.n_params)

params = np.random.randn(tcc_uccsd.n_params)
energy = tcc_uccsd.energy(params)
print(params)
print(energy)

assert energy == -0.6111608579235854
