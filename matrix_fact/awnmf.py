# Authors: Rikk Hill
# License: BSD 3 Clause
"""
MatrixFact Non-negative Matrix Factorization.

    AWNMF: Class for Augmented Weighted Non-negative Matrix Factorization

A horrible mess devised by Rikk
"""
import numpy as np
from .base import MatrixFactBase
import scipy

__all__ = ["AWNMF"]


class AWNMF(MatrixFactBase):
    """
    AWNMF(data, weights, num_bases=4)

    Non-negative Matrix Factorization. Factorize a data matrix into two matrices
    s.t. F = | S (*) (data - W*H) | is minimal. H, and W are restricted to non-negative
    data. S is a weighting matrix and (*) is Hadamard/elementwise multiplication.
    Uses the classicial multiplicative update rule.

    # (todo) Document this properly
    """
    def __init__(self, data, S, w_augments, h_augments=None, num_bases=4,
                 mask_zeros=False, **kwargs):
        MatrixFactBase.__init__(self, data, num_bases, **kwargs)

        if mask_zeros:
            mask = (data != 0).astype(int)
        else:
            mask = np.ones(data.shape)
        self.S = S * mask
        self.S_sqrt = np.sqrt(S)
        self.comp_S = (S - 1) * -1
        self.w_augments = w_augments
        self.h_augments = h_augments

        S_shape = S.shape

        # Make sure augments have the right dimensions
        assert w_augments.shape[0] == S_shape[0]
        assert w_augments.shape[1] <= num_bases


        # set w_augments index

        m_range_w = list(range(0, S_shape[0]))
        n_range_w = list(range(num_bases - w_augments.shape[1],
                               num_bases))
        self.w_augments_idx = np.ix_(m_range_w, n_range_w)

        # set h_augments index

        #n_range_h = list(range(0, S_shape[1]))
        #m_range_h = list(range(num_bases - h_augments.shape[1], num_bases))
        #self.h_augments_idx = np.ix_(m_range_h, n_range_h)

    def _update_h(self):
        # pre init H1, and H2 (necessary for storing matrices on disk)
        H2 = np.dot(self.W.T, self.S_sqrt * np.dot(self.W, self.H)) + 10**-9
        self.H *= np.dot(self.W.T, self.S_sqrt * self.data[:, :])
        self.H /= H2

        # Replace last m - n rows of H with augments
        #self.H[self.h_augments_idx] = self.h_augments.T

    def _update_w(self):
        # pre init W1, and W2 (necessary for storing matrices on disk)
        W2 = np.dot(self.S_sqrt * np.dot(self.W, self.H), self.H.T) + 10**-9
        self.W *= np.dot(self.S_sqrt * self.data[:, :], self.H.T)
        self.W /= W2
        self.W /= np.sqrt(np.sum(self.W ** 2.0, axis=0))

        # Replace last n rows of W with augments
        self.W[self.w_augments_idx] = self.w_augments

    def frobenius_norm(self, complement=False):
        """ Frobenius norm (||S (*)  (data - WH) ||) of a data matrix and a low rank
        approximation given by WH, weighted by S.

        If complement = True, this will return this value weighted by (S-1)*-1

        Parameters
        ----------
        complement : bool
            If true, return F_norm weighted by complement of weight matrix

        Returns:
        -------
        frobenius norm: F = || S (*) (data - WH)||

        Needs redefining for WNMF

        """

        if complement:
            S = self.comp_S
        else:
            S = self.S

        # check if W and H exist
        if hasattr(self,'H') and hasattr(self,'W'):
            if scipy.sparse.issparse(self.data):
                tmp = S * ( self.data[:,:] - (self.W * self.H) )
                tmp = tmp.multiply(tmp).sum()
                err = np.sqrt(tmp)
            else:
                err = np.sqrt(np.sum((S * (self.data[:, :] - np.dot(self.W, self.H))) ** 2))
        else:
            err = None

        return err


def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()