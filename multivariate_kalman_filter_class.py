import numpy as np
 import scipy.linalg as linalg
 import matplotlib.pyplot as plt
 import numpy.random as random
 from numpy import dot

class KalmanFilter:
    def __init__(self, dim_x, dim_z, dim_u=0):
        """
        Creates a Kalman filter
        :param dim_x: Number of state variables for the Kalman filter
        :param dim_z: Number of of measurement inputs.
        :param dim_u: size of the control input, if it is being used. Default = 0
        """

        self.x = np.zeros((dim_x, 1)) # state
        self.P = np.eye(dim_x) # uncertainty covariance
        self.Q = np.eye(dim_x) # process uncertainty
        self.u = np.zeros((dim_x, 1)) # motion vector
        self.B = 0 # control transition matrix
        self.F = 0 # state transition matrix
        self.H = 0 # Measurement function
        self.R = np.eye(dim_z) # state uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

    def predict(self, u=0):
        """ Predict next position. Parameters
        ----------
        u : np.array
        Optional control vector. If non-zero, it is multiplied by B
        to create the control input into the system.
        """

        self.x = dot(self.F, self.x) + dot(self.B, u)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def update(self, Z, R=None):
        """
        Add a new measurement (Z) to the kalman filter. If Z is None, nothing
        is changed.
        Parameters
        ----------
        Z : np.array
            measurement for this update.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        """
         if Z is None:
             return
         if R is None:
             R = self.R
         elif np.isscalar(R):
             R = np.eye(self.dim_z) * R

        # error (residual) between measurement and prediction
        y = Z - dot(H, x)

        # project system uncertainty into measurement space
        S = dot(H, P).dot(H.T) + R

        # map system uncertainty into kalman gain
        K = dot(P, H.T).dot(linalg.inv(S))

        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(K, y)

        # predict new system uncertainty
        I_KH = self._I - dot (K, H)
        self.P = dot(I_KH).dot(P).dot(I_KH.T) + dot(K, R).dot(K.T)


