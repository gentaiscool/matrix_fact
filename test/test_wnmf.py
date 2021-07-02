from matrix_fact.wnmf import *
from matrix_fact.nmf import NMF
import numpy as np
from numpy.testing import *
from base import *


class TestWNMF():

    data1 = np.array([[0.0, 0.0, 0.2],
                      [0.0, -1.0, 0.0]])

    data2 = np.array([[1.0, 0.0, 0.2],
                      [0.0, -1.0, 0.3]])

    weights = np.array([[0.0, 1.0, 1.0],
                        [1.0, 1.0, 0.0]])

    def test_wnmf(self):
        mdl1 = NMF(self.data1, num_bases=2)
        mdl2 = WNMF(self.data2, self.weights, num_bases=2)

        # NMF for data1 should perform identically to WNMF for data2
        mdl1.factorize(niter=1000)
        mdl2.factorize(niter=1000)

        assert(np.array_equal(mdl1.W, mdl2.W))
        assert(np.array_equal(mdl1.H, mdl2.H))
        assert(1 == 0)

        # W and H constrained to be > 0
        lh = np.where(mdl2.H < 0)[0]
        lw = np.where(mdl2.W < 0)[0]
        assert(len(lh) == 0)
        assert(len(lw) == 0)
