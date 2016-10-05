""" This file defines a Gaussian mixture model class. """
import logging

import numpy as np
import scipy.linalg

# TODO: rewrite this so we don't have such a thin wrapper around sklearn GMM.
from sklearn.mixture import GMM as sklearn_GMM

LOGGER = logging.getLogger(__name__)


def logsum(vec, axis=0, keepdims=True):
    #TODO: Add a docstring.
    maxv = np.max(vec, axis=axis, keepdims=keepdims)
    maxv[maxv == -float('inf')] = 0
    return np.log(np.sum(np.exp(vec-maxv), axis=axis, keepdims=keepdims)) + maxv

class GMM(object):
    """ Gaussian Mixture Model. """
    def __init__(self, init_sequential=False, eigreg=False, warmstart=True):
        self.init_sequential = init_sequential
        self.eigreg = eigreg
        self.warmstart = warmstart
        self.sigma = None
        self.gmm = None

    def inference(self, pts):
        """
        Evaluate dynamics prior.
        Args:
            pts: A N x D array of points.
        """
        # Compute posterior cluster weights.
        wts = self.clusterwts(pts)

        # Compute posterior mean and covariance.
        mu0, Phi = self.moments(wts)

        # Set hyperparameters.
        m = self.N
        n0 = m - 2 - mu0.shape[0]

        # Normalize.
        m = float(m) / self.N
        n0 = float(n0) / self.N
        return mu0, Phi, m, n0

    def moments(self, wts):
        """
        Compute the moments of the cluster mixture with logwts.
        Args:
            wts: A K x 1 array of cluster probabilities.
        Returns:
            mu: A (D,) mean vector.
            sigma: A D x D covariance matrix.
        """
        # Exponentiate.
        mus = self.gmm.means_     # (K, D)
        sigmas = self.gmm.covars_ # (K, D, D)

        # Compute overall mean.
        mu = np.sum(mus * wts, axis=0)

        # Compute overall covariance.
        # For some reason this version works way better than the "right"
        # one... could we be computing xxt wrong?
        diff = mus - np.expand_dims(mu, axis=0)
        diff_expand = np.expand_dims(diff, axis=1) * \
                np.expand_dims(diff, axis=2)
        wts_expand = np.expand_dims(wts, axis=2)
        sigma = np.sum((sigmas + diff_expand) * wts_expand, axis=0)
        return mu, sigma

    def clusterwts(self, data):
        """
        Compute cluster weights for specified points under GMM.
        Args:
            data: An (N, D) array of points
        Returns:
            A (K,) array of average cluster probabilities.
        """
        prob = self.gmm.predict_proba(data)  # (N, K)
        return prob.mean(axis=0)[:, None]    # (K, 1)

    def update(self, data, K, max_iterations=100):
        """
        Run EM to update clusters.
        Args:
            data: An N x D data matrix, where N = number of data points.
            K: Number of clusters to use.
        """
        # Constants.
        N = data.shape[0]
        print('Fitting GMM with %d clusters on %d points', K, N)

        # TODO: rewrite this class so we don't have such a thin wrapper.
        # if self.gmm is None:
        self.gmm = sklearn_GMM(n_components=K, n_iter=100,
                init_params='', covariance_type='full', tol=0.01)
        # elif K != self.gmm.weights_.size:
        #     self.gmm.set_params(n_components=K)

        self.N = N
        self.gmm.fit(data)
