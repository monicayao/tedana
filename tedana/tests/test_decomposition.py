"""
Tests for tedana.decomposition
"""

import pytest
import numpy as np
from random import sample, randint 

from tedana import decomposition

# SMOKE TEST 

# _utils.py 
def test_smoke_eimask():
    n_samples, n_echos, n_times= 100, 5, 20
    dd = np.random.random((n_samples, n_echos, n_times))
    # ees = sample(list(range(0, n_echos)), randint(0, n_echos))
    # ees = [1, 2, 3]
    assert decomposition._utils.eimask(dd) is not None


# ica.py
def test_smoke_tedica():
    n_samples, n_times, n_components = 100, 20, 6
    data = np.random.random((n_samples, n_times)) 
    fixed_seed, maxit, maxrestart = randint(0, 10), randint(3, 10), randint(3, 10) # keep these small so test is faster

    assert decomposition.tedica(data, n_components, fixed_seed, maxit, maxrestart) is not None


# pca.py
def test_smoke_run_mlepca():
    n_samples, n_echos, n_times = 100, 5, 20
    data_2d = np.random.random((n_samples, n_times))
    #data_3d = np.random.random((n_samples, n_echos, n_times))  

    u_2d, s_2d, varex_norm_2d, v_2d = decomposition.pca.run_mlepca(data_2d)

    assert all(v is not None for v in [u_2d, s_2d, varex_norm_2d, v_2d])


def test_smoke_low_mem_pca():
    n_samples, n_echos, n_times = 100, 5, 20
    data_2d = np.random.random((n_samples, n_times))
    #data_3d = np.random.random((n_samples, n_echos, n_times))  

    u_2d, s_2d, v_2d = decomposition.pca.low_mem_pca(data_2d)

    assert all(v is not None for v in [u_2d, s_2d, v_2d])

# TODO: def test_smoke_tedpca(): <---- this requires ref_img