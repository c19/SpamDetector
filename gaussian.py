#!/usr/bin/env python
# coding=utf-8
#    trainer
#    C19<caoyijun2050@gmail.com>

# from sklearn.covariance import EllipticEnvelope
from base import do
import numpy as np


def Probability(mean, covariance):
    n = len(mean)
    det_square = np.sqrt(np.linalg.det(covariance))
    cov_inverse = np.linalg.inv(covariance)
    coefficient = 1.0 / (np.power(2 * np.pi, n / 2.0) * det_square)
    return lambda x: coefficient * (np.dot(np.dot((x - mean), cov_inverse), (x - mean))) ** 2


# sadly, without curry, I have to trea every function as single paramed.
# ohterwise, provide the extra info would be a overhead.
def Gaussian(nparray):
    return do(nparray,
              lambda x: (x.mean(axis=0), np.cov(x.T)),
              lambda x: Probability(*x),
              )
