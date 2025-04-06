__version__ = "2023.03"
__author__ = "TenCirChem Authors"
__creator__ = "Weitang Li"

#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.

import os
import logging
import numpy as np

# for debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# disable CUDA 11.1 warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.getLogger("tensorcircuit")
logger.setLevel(logging.FATAL)

os.environ["RENO_LOG_LEVEL"] = "100"

logger = logging.getLogger("tencirchem")
logger.setLevel(logging.WARNING)

# finish logger stuff
del logger

# by default use float64 rather than float32
rdtypestr = "float64"
if rdtypestr == "float64":
    uint_type = np.uint64
else:
    assert rdtypestr == "float32"
    uint_type = np.uint32

# static module
# as an external interface
from pyscf import M

from tencirchem import hamiltonian
import tencirchem.ucc as ucc

# from tencirchem.static.ucc import UCC
# from tencirchem.static.uccsd import UCCSD, ROUCCSD


def clear_cache():
    from .evolve_civector import CI_OPERATOR_CACHE, CI_OPERATOR_BATCH_CACHE

    CI_OPERATOR_CACHE.clear()
    CI_OPERATOR_BATCH_CACHE.clear()
