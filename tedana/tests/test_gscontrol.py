"""
Tests for tedana.model.fit
"""

import numpy as np
import pytest

import tedana.gscontrol as gsc


def test_break_gscontrol_raw():
    """
    Ensure that gscontrol_raw fails when input data do not have the right
    shapes.
    """
    n_samples, n_echos, n_vols = 10000, 4, 100
    catd = np.empty((n_samples, n_echos, n_vols))
    optcom = np.empty((n_samples, n_vols))
    ref_img = ''

    catd = np.empty((n_samples+1, n_echos, n_vols))
    with pytest.raises(ValueError) as e_info:
        gsc.gscontrol_raw(catd=catd, optcom=optcom, n_echos=n_echos,
                          ref_img=ref_img, dtrank=4)
    assert str(e_info.value) == ('First dimensions of catd ({0}) and optcom ({1}) do not '
                                 'match'.format(catd.shape[0], optcom.shape[0]))

    catd = np.empty((n_samples, n_echos+1, n_vols))
    with pytest.raises(ValueError) as e_info:
        gsc.gscontrol_raw(catd=catd, optcom=optcom, n_echos=n_echos,
                          ref_img=ref_img, dtrank=4)
    assert str(e_info.value) == ('Second dimension of catd ({0}) does not match '
                                 'n_echos ({1})'.format(catd.shape[1], n_echos))

    catd = np.empty((n_samples, n_echos, n_vols))
    optcom = np.empty((n_samples, n_vols+1))
    with pytest.raises(ValueError) as e_info:
        gsc.gscontrol_raw(catd=catd, optcom=optcom, n_echos=n_echos,
                          ref_img=ref_img, dtrank=4)
    assert str(e_info.value) == ('Third dimension of catd ({0}) does not match '
                                 'second dimension of optcom '
                                 '({1})'.format(catd.shape[2], optcom.shape[1]))


# SMOKE TEST

def test_smoke_gscontrol_raw():
    n_samples, n_times, n_echos = 64350, 10, 1
    catd = np.random.random((n_samples, n_echos, n_times))
    optcom = np.random.random((n_samples, n_times))
    ref_img = "data/mask.nii.gz" 

    assert gsc.gscontrol_raw(catd, optcom, n_echos, ref_img)


# def test_smoke_gscontrol_mmix(): test without no return value  
#   return 