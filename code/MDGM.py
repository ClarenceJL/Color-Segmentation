"""
Author: Claire Li
Date: Jan, 2018
"""
import numpy as np

from scipy.linalg import eigh

def log_pdf(X, mu, Sigma):
    """
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    """

    d = np.shape(X)[1]
    S, U = eigh(Sigma, lower=True, check_finite=True)
    eps = np.max(abs(S)) * 2e-10

    if np.min(S) < eps:
        raise ValueError('Covariance matrix is not positive semi-definite')

    S_pinv = np.array([0 if abs(x) < 1e-5 else 1/x for x in S], dtype=float)
    Sigma_pinv_sqrt = np.multiply(U, np.sqrt(S_pinv))
    log_det_Sigma = np.sum(np.log(S[S > eps]))

    Xc = X - mu
    Y = np.sum(np.square(np.dot(Xc, Sigma_pinv_sqrt)), axis=1)

    log_prob = -0.5 * (d * np.log(np.pi * 2) + log_det_Sigma + Y)

    return log_prob


def multivariate_Gaussian_pdf(X, mu, Sigma):
    """
    Calculate the probability of data samples conforming to
    the given multivariate Gaussian (Normal) distribution
    :param X: [n, d]
    :param mu: [d]
    :param Sigma: [d, d]
    :return: prob[d]
    """

    log_prob = log_pdf(X, mu, Sigma)
    prob = np.exp(log_prob)

    return prob




class MD_Gaussian_Model(object):
    """
    This model using two single Gaussian Distribution to represent
    the barrel red color model and the background color model respectively
    """
    def __init__(self, X_pos_train, X_neg_train):
        # data
        self.X_red_train = X_pos_train.astype(float)
        self.X_bg_train = X_neg_train.astype(float)
        d = np.shape(X_pos_train)[1]

        # model parameters (to be learned)
        # prior: probability of a pixel being positive
        self.prior_pos = 0.5
        # likelihood:
        self.mu_red = np.zeros([d]) + 120
        self.A_red = 100*np.identity(d)
        self.mu_bg = np.zeros([d]) + 80
        self.A_bg = 500*np.identity(d)

        # train model
        self.training()

    def training(self):
        # set prior according to the statistic property of training data
        self.prior_pos = np.size(self.X_red_train, axis=0) / \
                         (np.size(self.X_red_train, axis=0) + np.size(self.X_bg_train, axis=0))

        # MLE for positive model
        self.mu_red = np.mean(self.X_red_train, axis=0)
        self.A_res = np.cov(np.transpose(self.X_red_train))

        # MLE for negative model
        self.mu_bg = np.mean(self.X_bg_train, axis=0)
        self.A_bg = np.cov(np.transpose(self.X_bg_train))

    def inference(self, X_test):
        """
        :param X_test: [n, d]
        :return: labels, [n,]
        """
        likelihood_red = multivariate_Gaussian_pdf(X_test, self.mu_red, self.A_red)
        likelihood_bg = multivariate_Gaussian_pdf(X_test, self.mu_bg, self.A_bg)
        prob = likelihood_red * self.prior_pos / (likelihood_red * self.prior_pos + likelihood_bg * (1-self.prior_pos))
        labels = (prob >= 0.5)

        return prob, labels


