"""
Tests for tedana.metrics.fit
"""

import numpy as np
import pytest

from tedana.metrics import kundu_fit


def test_break_dependence_metrics():
    """
    Ensure that dependence_metrics fails when input data do not have the right
    shapes.
    """
    n_samples, n_echos, n_vols, n_comps = 10000, 4, 100, 50
    catd = np.empty((n_samples, n_echos, n_vols))
    tsoc = np.empty((n_samples, n_vols))
    mmix = np.empty((n_vols, n_comps))
    t2s = np.empty((n_samples, n_vols))
    t2s_full = np.empty((n_samples, n_vols))
    tes = np.empty((n_echos))
    combmode = 't2s'
    ref_img = ''

    # Shape of catd is wrong
    catd = np.empty((n_samples+1, n_echos, n_vols))
    with pytest.raises(ValueError) as e_info:
        kundu_fit.dependence_metrics(catd=catd, tsoc=tsoc, mmix=mmix,
                               t2s=t2s, tes=tes, ref_img=ref_img,
                               reindex=False, mmixN=None, algorithm='kundu_v3')
    assert str(e_info.value) == ('First dimensions (number of samples) of '
                                 'catd ({0}), tsoc ({1}), '
                                 'and t2s ({2}) do not match'.format(
                                    catd.shape[0], tsoc.shape[0],
                                    t2s.shape[0]))

    # Shape of t2s is wrong
    catd = np.empty((n_samples, n_echos, n_vols))
    t2s = np.empty((n_samples+1, n_vols))
    with pytest.raises(ValueError) as e_info:
        kundu_fit.dependence_metrics(catd=catd, tsoc=tsoc, mmix=mmix,
                               t2s=t2s, tes=tes, ref_img=ref_img,
                               reindex=False, mmixN=None, algorithm='kundu_v3')
    assert str(e_info.value) == ('First dimensions (number of samples) of '
                                 'catd ({0}), tsoc ({1}), '
                                 'and t2s ({2}) do not match'.format(
                                    catd.shape[0], tsoc.shape[0],
                                    t2s.shape[0]))

    # Shape of tsoc is wrong
    t2s = np.empty((n_samples, n_vols))
    tsoc = np.empty((n_samples+1, n_vols))
    with pytest.raises(ValueError) as e_info:
        kundu_fit.dependence_metrics(catd=catd, tsoc=tsoc, mmix=mmix,
                               t2s=t2s, tes=tes, ref_img=ref_img,
                               reindex=False, mmixN=None, algorithm='kundu_v3')
    assert str(e_info.value) == ('First dimensions (number of samples) of '
                                 'catd ({0}), tsoc ({1}), '
                                 'and t2s ({2}) do not match'.format(
                                    catd.shape[0], tsoc.shape[0],
                                    t2s.shape[0]))

    # Shape of catd is wrong
    catd = np.empty((n_samples, n_echos+1, n_vols))
    tsoc = np.empty((n_samples, n_vols))
    with pytest.raises(ValueError) as e_info:
        kundu_fit.dependence_metrics(catd=catd, tsoc=tsoc, mmix=mmix,
                               t2s=t2s, tes=tes, ref_img=ref_img,
                               reindex=False, mmixN=None, algorithm='kundu_v3')
    assert str(e_info.value) == ('Second dimension of catd ({0}) does not '
                                 'match number '
                                 'of echoes provided (tes; '
                                 '{1})'.format(catd.shape[1], len(tes)))

    # Shape of catd is wrong
    catd = np.empty((n_samples, n_echos, n_vols+1))
    with pytest.raises(ValueError) as e_info:
        kundu_fit.dependence_metrics(catd=catd, tsoc=tsoc, mmix=mmix,
                               t2s=t2s, tes=tes, ref_img=ref_img,
                               reindex=False, mmixN=None, algorithm='kundu_v3')
    assert str(e_info.value) == ('Number of volumes in catd '
                                 '({0}), tsoc ({1}), and '
                                 'mmix ({2}) do not '
                                 'match.'.format(catd.shape[2], tsoc.shape[1],
                                                 mmix.shape[0]))

    # Shape of t2s is wrong
    catd = np.empty((n_samples, n_echos, n_vols))
    t2s = np.empty((n_samples, n_vols+1))
    with pytest.raises(ValueError) as e_info:
        kundu_fit.dependence_metrics(catd=catd, tsoc=tsoc, mmix=mmix,
                               t2s=t2s, tes=tes, ref_img=ref_img,
                               reindex=False, mmixN=None, algorithm='kundu_v3')
    assert str(e_info.value) == ('Number of volumes in catd ({0}) '
                                 'does not match number of volumes in '
                                 't2s ({1})'.format(catd.shape[2], t2s.shape[1]))


def test_smoke_dependence_metrics():
    n_samples, n_times, n_echos, n_components = 64350, 10, 1, 6
    catd = np.random.random((n_samples, n_echos, n_times))
    tsoc = np.random.random((n_samples, n_times))
    mmix = np.random.random((n_times, n_components))

    t2s = np.random.random((n_samples))
    tes = np.random.random((n_echos)).tolist()
    ref_img = "data/mask.nii.gz" 
    
    comptable, seldict, betas, mmix_new = kundu_fit.dependence_metrics(catd, tsoc, mmix, t2s, tes, ref_img, algorithm="kundu_v2")

    assert comptable is not None
    assert seldict is not None 
    assert betas is not None
    assert mmix_new is not None