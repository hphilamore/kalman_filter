from __future__ import print_function, division
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy.random as random
import numpy as np


class KalmanFilter1D:
    def __init__(self, x0, P, R, Q):
        self.x = x0 # state
        self.P = P # state variance
        self.R = R # measurement error
        self.Q = Q # movement error
    def update(self, z):
        """
        New state found by multiplying state and measurement gaussians
        :param z: sensor measurement
        """
        # mean state as a result of multiplication
        self.x = (self.P * z + self.x * self.R) / (self.P + self.R)
        # state variance as a result of multiplication
        self.P = 1. / (1./self.P + 1./self.R)
    def predict(self, u=0.0):
        """
        New state found by adding state and movement gaussians
        :param u:
        """
        # mean state as a result of addition
        self.x += u
        # state variance as a result of addition
        self.P += self.Q


