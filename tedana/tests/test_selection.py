"""
Tests for tedana.selection
"""

import os.path as op

import pytest
import numpy as np
import pandas as pd
from random import random 

from tedana import selection


def test_manual_selection():
    """
    Check that manual_selection runs correctly for different combinations of
    accepted and rejected components.
    """
    comptable = pd.DataFrame(index=np.arange(100))
    comptable = selection.manual_selection(comptable, acc=[1, 3, 5])
    assert comptable.loc[comptable.classification == 'accepted'].shape[0] == 3
    assert (comptable.loc[comptable.classification == 'rejected'].shape[0] ==
            (comptable.shape[0] - 3))

    comptable = selection.manual_selection(comptable, rej=[1, 3, 5])
    assert comptable.loc[comptable.classification == 'rejected'].shape[0] == 3
    assert (comptable.loc[comptable.classification == 'accepted'].shape[0] ==
            (comptable.shape[0] - 3))

    comptable = selection.manual_selection(comptable, acc=[0, 2, 4],
                                           rej=[1, 3, 5])
    assert comptable.loc[comptable.classification == 'accepted'].shape[0] == 3
    assert comptable.loc[comptable.classification == 'rejected'].shape[0] == 3
    assert (comptable.loc[comptable.classification == 'ignored'].shape[0] ==
            comptable.shape[0] - 6)


# SMOKE TESTS 

# _utils.py
def test_smoke_clean_dataframe():
    """ 
    Smoke test for the clean_dataframe function in selection._utils
    """
    # creating a dataframe with the correct columns but random data
    n_components = 100
    random_metric = np.random.random((n_components))
    classification = np.random.choice(["accepted", "rejected", "ignored"], n_components)
    rationale = np.random.choice(["P001;", "P002;", "I001", "I002;"], n_components)
    df_data = np.column_stack((random_metric, rationale, classification))
    comptable = pd.DataFrame(df_data, columns=['random_metric', 'rationale', 'classification'])
    assert selection._utils.clean_dataframe(comptable) is not None


def test_smoke_getelbow_cons():
    """
    Smoke test for getelbow_cons function in selection._utils
    """
    arr = np.random.random((100))

    assert selection._utils.getelbow_cons(arr) is not None
    assert selection._utils.getelbow_cons(arr, return_val=True) is not None


def test_getelbow():
    arr = np.random.random((100))

    assert selection._utils.getelbow(arr) is not None
    assert selection._utils.getelbow(arr, return_val=True) is not None


# tedica.py
def test_smoke_manual_selection():
    n_components = 100
    
    # create a dataframe of component and random accepted/rejected
    random_metric = np.random.random((n_components))
    classification = np.random.choice(["accepted", "rejected", "ignored"], n_components)
    rationale = np.random.choice(["P001", "P002", "I001", "I002"], n_components)
    df_data = np.column_stack((random_metric, classification, rationale))
    comptable = pd.DataFrame(df_data, columns=['component', 'classification', 'rationale'])

    acc = np.random.randint(2, size=n_components)
    rej = np.random.randint(2, size=n_components)

    assert selection.manual_selection(comptable) is not None
    assert selection.manual_selection(comptable, acc=acc) is not None
    assert selection.manual_selection(comptable, rej=rej) is not None


def test_smoke_kundu_selection_v2():
    n_components = 100

    # create a dataframe of metrics needed for ica 
    comptable = pd.DataFrame(columns=['kappa', 'rho', 'variance explained',
                                      'normalized variance explained', 'countsigFS0', 'countsigFR2',
                                      'F_R2_clmaps', 'F_S0_clmaps', 'dice_FS0', 'dice_FR2', 'signal-noise_t',
                                      'd_table_score', 'countnoise', 'd_table_score_scrub'],
                             data=np.random.random((n_components, 14)),
                             index=np.arange(n_components))

    n_echos, n_vols = 5, 10

    assert selection.kundu_selection_v2(comptable, n_echos, n_vols) is not None

# tedpca.py
def test_smoke_kundu_tedpca():
    comptable = pd.DataFrame(columns=['kappa', 'rho', 'variance explained',
                                      'normalized variance explained'],
                             data=np.random.random((100, 4)),
                             index=np.arange(100))

    n_echos = 5
    kdaw, rdaw = random(), random()

    assert selection.kundu_tedpca(comptable, n_echos) is not None
    assert selection.kundu_tedpca(comptable, n_echos, kdaw=kdaw) is not None
    assert selection.kundu_tedpca(comptable, n_echos, rdaw=rdaw) is not None
    assert selection.kundu_tedpca(comptable, n_echos, stabilize=True) is not None
    
