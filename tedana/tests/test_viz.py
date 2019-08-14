import numpy as np

from tedana import viz

def test_smoke_trim_edge_zeros():
    arr = np.random.random((100, 100))

    assert viz.trim_edge_zeros(arr) is not None