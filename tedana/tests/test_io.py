"""
Tests for tedana.io
"""

import nibabel as nib
import numpy as np
import pytest
import pandas as pd

from tedana import io as me
from tedana.tests.test_utils import fnames, tes


def test_new_nii_like():
    data, ref = me.load_data(fnames, n_echos=len(tes))
    nimg = me.new_nii_like(ref, data)

    assert isinstance(nimg, nib.Nifti1Image)
    assert nimg.shape == (39, 50, 33, 3, 5)


def test_filewrite():
    pass


def test_load_data():
    fimg = [nib.load(f) for f in fnames]
    exp_shape = (64350, 3, 5)

    # list of filepath to images
    d, ref = me.load_data(fnames, n_echos=len(tes))
    assert d.shape == exp_shape
    assert isinstance(ref, nib.Nifti1Image)
    assert np.allclose(ref.get_data(), nib.load(fnames[0]).get_data())

    # list of img_like
    d, ref = me.load_data(fimg, n_echos=len(tes))
    assert d.shape == exp_shape
    assert isinstance(ref, nib.Nifti1Image)
    assert ref == fimg[0]

    # imagine z-cat img
    d, ref = me.load_data(fnames[0], n_echos=3)
    assert d.shape == (21450, 3, 5)
    assert isinstance(ref, nib.Nifti1Image)
    assert ref.shape == (39, 50, 11, 1)

    with pytest.raises(ValueError):
        me.load_data(fnames[0])


# SMOKE TESTS

def test_smoke_split_ts():
    """
    Note: classification is ["accepted", "rejected", "ignored"]
    """
    n_samples = 100 
    n_times = 20
    n_components = 6
    data = np.random.random((n_samples, n_times))
    mmix = np.random.random((n_times, n_components))
    mask = np.random.randint(2, size=n_samples)
    
    # creating the component table with component as random floats and random classification
    components = np.random.random((n_components))
    classification = np.random.choice(["accepted", "rejected", "ignored"], n_components)
    df_data = np.column_stack((components, classification))
    comptable = pd.DataFrame(df_data, columns=['component', 'classification'])

    hikts, resid = me.split_ts(data, mmix, mask, comptable)

    assert hikts is not None
    assert resid is not None


def test_smoke_write_split_ts(): # TODO because of the ref_img
    """ can't do it with the img """ 
    n_samples, n_times, n_components = 100, 20, 6
    data = np.random.random((n_samples, n_times))
    mmix = np.random.random((n_times, n_components))
    mask = np.random.randint(2, size=n_samples)
    ref_img = ''

    # creating the component table with component as random floats and random classification
    components = np.random.random((n_components))
    classification = np.random.choice(["accepted", "rejected", "ignored"], n_components)
    df_data = np.column_stack((components, classification))
    comptable = pd.DataFrame(df_data, columns=['component', 'classification'])

    assert me.write_split_ts(data, mmix, mask, comptable, ref_img) is not None


def test_smoke_writefeats():
    return


def test_smoke_writeresults():
    return 


def test_smoke_new_nii_like():
    return


def test_smoke_filewrite():
    return


def test_smoke_load_data():
    """ problem with check_niimg(data) """
    data = np.random.random((100, 20, 10, 5)) # randomly shaped ME array
    n_echos = 10

    fdata, ref_img = me.load_data(data, n_echos)
    assert fdata is not None
    assert ref_img is not None

# TODO: "BREAK" AND UNIT TESTS
