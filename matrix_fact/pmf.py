# Authors: Rikk Hill
# License: BSD 3 Clause
"""
MatrixFact Non-negative Matrix Factorization.

    WNMF: Class for Weighted Non-negative Matrix Factorization

[1] Lee, D. D. and Seung, H. S. (1999), Learning the Parts of Objects by Non-negative
Matrix Factorization, Nature 401(6755), 788-799.
"""
import numpy as np
from .base import MatrixFactBase
import scipy
from scipy import special

__all__ = ["PMF"]


class PMF(MatrixFactBase):
    """
        Poisson Matrix Factorisation
        Variational Bayesian factorisation
    """
    def __init__(self, data, num_bases=4, augments=None, smoothness=100, **kwargs):


        # Setup
        self.num_bases = num_bases
        self.smoothness = smoothness
        data_shape = data.shape
        self.w_outer = data_shape[0]
        self.h_outer = data_shape[1]

        self.augments = augments

        if self.augments is not None:
            # Make sure augments have the right dimensions
            assert augments.shape[0] == data.shape[0]
            assert augments.shape[1] <= num_bases

            m_range = list(range(0, data.shape[0]))

            n_range = list(range(num_bases - augments.shape[1], num_bases))

            self.augments_idx = np.ix_(m_range, n_range)

        MatrixFactBase.__init__(self, data, num_bases, **kwargs)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.a = float(kwargs.get('a', 0.1))
        self.b = float(kwargs.get('b', 0.1))

    def _init_h(self):
        # variational parameters for beta / H
        self.gamma_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.num_bases, self.h_outer))
        self.rho_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.num_bases, self.h_outer))
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def _init_w(self):
        # variational parameters for theta
        self.gamma_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.w_outer, self.num_bases))
        self.rho_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.w_outer, self.num_bases))
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)
        self.c = 1. / np.mean(self.Et)

    def _update_w(self):
        ratio = self.data / self._xexplog()
        self.gamma_t = self.a + np.exp(self.Elogt) * np.dot(
            ratio, np.exp(self.Elogb).T)
        self.rho_t = self.a * self.c + np.sum(self.Eb, axis=1)
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)
        self.c = 1. / np.mean(self.Et)

        # Replace last n rows of W with augments
        if self.augments is not None:
            self.Et[self.augments_idx] = self.augments

    def _update_h(self):
        ratio = self.data / self._xexplog()
        self.gamma_b = self.b + np.exp(self.Elogb) * np.dot(
            np.exp(self.Elogt).T, ratio)
        self.rho_b = self.b + np.sum(self.Et, axis=0, keepdims=True).T
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def _xexplog(self):
        '''
        sum_k exp(E[log theta_{ik} * beta_{kd}])
        '''
        return np.dot(np.exp(self.Elogt), np.exp(self.Elogb))

    def _bound(self):
        bound = np.sum(self.data * np.log(self._xexplog()) - self.Et.dot(self.Eb))
        bound += _gamma_term(self.a, self.a * self.c,
                             self.gamma_t, self.rho_t,
                             self.Et, self.Elogt)
        bound += self.num_bases * self.data.shape[0] * self.a * np.log(self.c)
        bound += _gamma_term(self.b, self.b, self.gamma_b, self.rho_b,
                             self.Eb, self.Elogb)
        return bound

    def frobenius_norm(self, complement=False):

        # check if W and H exist
        if hasattr(self, 'Eb') and hasattr(self, 'Et'):
            if scipy.sparse.issparse(self.data):
                tmp = (self.data[:, :] - (self.Et * self.Eb))
                tmp = tmp.multiply(tmp).sum()
                err = np.sqrt(tmp)
            else:
                err = np.sqrt(np.sum((self.data[:, :] - np.dot(self.Et, self.Eb)) ** 2))
        else:
            err = None

        return err


def _compute_expectations(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute E[x] and E[log x]
    '''
    return alpha / beta, special.psi(alpha) - np.log(beta)


def _gamma_term(a, b, shape, rate, Ex, Elogx):
    return np.sum((a - shape) * Elogx - (b - rate) * Ex +
                  (special.gammaln(shape) - shape * np.log(rate)))


def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()