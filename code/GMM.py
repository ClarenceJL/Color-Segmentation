"""
GMM using EM training algorithm
Author: Claire Li
Date: Jan, 2018
"""
import numpy as np
from MDGM import multivariate_Gaussian_pdf, log_pdf


def expectation(X, mu_set, Sigma_set, Pi):
    """
    :param X: [n, d] training dataset
    :param mu_set: [K, d]
    :param Sigma_set: [K, d, d]
    :param Pi_set: [K,]
    :return: posterior [n, K]
    """
    n = np.shape(X)[0]
    K = np.shape(mu_set)[0]
    posterior = np.zeros([n, K])
    for i in range(K):
        likelihood = multivariate_Gaussian_pdf(X, mu_set[i, :], Sigma_set[i, :, :])
        posterior[:, i] = Pi[i] * likelihood

    # normalization
    postsum = posterior.sum(axis=1)  # sum over K Gaussians
    posterior = posterior / postsum[:, np.newaxis] # each row sum to 1

    return posterior


def maximization(X, posterior):
    """
    update parameters
    :param X: [n, d]
    :param posterior: [n, K]
    :return:
    """
    n, d = np.shape(X)
    K = np.shape(posterior)[1]

    # update pi
    Pi = np.reshape(np.mean(posterior, axis=0), [-1])
    Pi = Pi / np.sum(Pi)

    # normalize posterior by column
    post_sum = posterior.sum(axis=0)
    post_norm = posterior / post_sum[np.newaxis, :]

    # update mu & Sigma
    mu = np.matmul(np.transpose(post_norm), X)

    Sigma = np.zeros([K, d, d])
    for k in range(K):
        p_k = np.reshape(post_norm[:, k],[-1])
        X_c = X - mu[k, :]
        C_k = np.matmul(np.transpose(X_c), np.multiply(X_c, p_k[:,np.newaxis]))
        Sigma[k, :, :] = C_k

    return mu, Sigma, Pi


def terminate_condition_theta(mu1, Sigma1, Pi1, mu2, Sigma2, Pi2, eps = 5e-4):
    theta1 = np.concatenate((np.reshape(mu1, [-1]), np.reshape(Sigma1, [-1]), np.reshape(Pi1, [-1])))
    theta2 = np.concatenate((np.reshape(mu2, [-1]), np.reshape(Sigma2, [-1]), np.reshape(Pi2, [-1])))
    dist = np.linalg.norm(theta1 - theta2)
    return dist <= eps


def terminate_condition_log_prob(prob_prev, prob_new, eps=1e-6):
    # compute the KL-divergence of previous and new distribution
    prob_prev[prob_prev < 1e-8] = 1e-8
    prob_new[prob_new < 1e-8] = 1e-8
    log_prev = np.log(prob_prev) #q
    log_new = np.log(prob_new) # q
    KL_div = np.sum(np.multiply(prob_new, log_new - log_prev), axis=1)
    dist = np.mean(KL_div)
    return dist <= eps


def gmm_probability(X, mu_set, Sigma_set, Pi):
    """
    :param X: [n, d]
    :param mu_set: [K, d]
    :param Sigma_set: [K, d, d]
    :param Pi: [K,]
    :return:
    """
    n, d = np.shape(X)
    K = np.shape(mu_set)[0]
    prob = np.zeros([n])

    for k in range(K):
        likelihood = multivariate_Gaussian_pdf(X, mu_set[k, :], Sigma_set[k, :, :])
        prob = prob + likelihood * Pi[k]

    return prob


class GaussianMixtureModel(object):
    """
    This model using two Gaussian Mixture Models to represent
    the barrel red color model and the background color model respectively.
    """
    def __init__(self, X_pos_train, X_neg_train, threshold=-12, K_red=3, K_bg=6):
        # data
        self.X_red_train = np.asarray(X_pos_train, dtype=float)
        self.X_bg_train = np.asarray(X_neg_train, dtype=float)
        self.max_iter = 200
        self.d = np.shape(X_pos_train)[1]

        pos_size = np.shape(X_pos_train)[0]
        neg_size = np.shape(X_neg_train)[0]

        # model parameters
        self.threshold = threshold
        self.prior_pos = 0.5
        I_d = np.reshape(100*np.identity(self.d), [1, self.d, self.d])
        # parameter for positive color class
        self.K_red = K_red
        self.post_red = np.zeros([X_pos_train.shape[0], self.K_red])
        # randomly choose initial means
        self.Mu_red = self.X_red_train[np.random.choice(pos_size, self.K_red, False), :]
        self.Sigma_red = np.tile(I_d, [self.K_red, 1, 1])
        self.Pi_red = np.ones([self.K_red])/self.K_red

        # parameter for negative color class
        self.mu_bg = np.zeros([self.d]) + 80
        self.A_bg = 500*np.identity(self.d)

        # train model
        self.training_em()

    def training_em(self):
        """
        Use EM algorithm to train the parameters for the positive and negative GMM models
        :return:
        """
        # set prior according to the statistic property of training data
        self.prior_pos = np.size(self.X_red_train, axis=0) / \
                         (np.size(self.X_red_train, axis=0) + np.size(self.X_bg_train, axis=0))

        # train positive samples
        print('training positive model ...')
        prev_prob = np.sum(np.log(np.zeros([self.X_red_train.shape[0]])+1e-40))
        for i in range(self.max_iter):
            if i % 10 == 0:
                print('checkpoint {}'.format(int(i/10)))
            prev_mu = self.Mu_red
            prev_Sigma = self.Sigma_red
            prev_pi = self.Pi_red
            # E-step
            self.post_red = expectation(self.X_red_train, self.Mu_red, self.Sigma_red, self.Pi_red)
            # M-step
            self.Mu_red, self.Sigma_red, self.Pi_red = maximization(self.X_red_train, self.post_red)
            # pre-terminating
            #if terminate_condition_theta(prev_mu, prev_Sigma, prev_pi, self.Mu_red, self.Sigma_red, self.Pi_red):
            #    break
            prob = np.sum(np.log(gmm_probability(self.X_red_train, self.Mu_red, self.Sigma_red, self.Pi_red)))
            print('log likelihood {}'.format(prob))

            if abs(prev_prob-prob) < 1:
                break
            prev_prob = prob

        # train negative samples

        self.mu_bg = np.mean(self.X_bg_train, axis=0)
        self.A_bg = np.cov(np.transpose(self.X_bg_train))

        # clear the training data after training, but save the variables as a record
        self.X_red_train = np.empty([1])
        self.X_bg_train = np.empty([1])

    def inference(self, X_test):
        """
        :param X_test: [n, d]
        :return: labels, [n,]
        """
        X_test = np.asarray(X_test, dtype=float)
        likelihood_red = gmm_probability(X_test, self.Mu_red, self.Sigma_red, self.Pi_red)
        likelihood_bg = multivariate_Gaussian_pdf(X_test, self.mu_bg, self.A_bg)
        prob = likelihood_red * self.prior_pos / (likelihood_red * self.prior_pos + likelihood_bg * (1-self.prior_pos))
        labels = (prob >= 0.6)

        return prob, labels
