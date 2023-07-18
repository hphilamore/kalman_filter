from __future__ import print_function, division
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy.random as random
import numpy as np
from kalman_filter_class import KalmanFilter1D
from time import sleep

def volt(temp_variance):
    """
    Simulates the voltage of a temperature sensor = 16.3 V
    with white noice charcterised by the variance specified
    :param temp_variance:
    :return:
    """
    return random.randn()*temp_variance + 16.3

temp_SD = 2.13
temp_variance = temp_SD**2
sensor_error = temp_variance
movement = 0                                 # constant temperature
movement_error = .2                          # expected variance in change of voltage (avoids smug filter)
N=50                                         # number of time steps
zs = [volt(temp_variance) for i in range(N)] # generate array of sensor values
state_estimate_x = [] # array to store series of positions
state_variance_P = [] # array to store series of position variance
# estimates = []

kf = KalmanFilter1D(x0=25,            # initial state estimate
                    P = 1000,         # initial variance - large initial estimate due to uncertainty in true value
                    R=temp_variance,  # sensor noise
                    Q=movement_error) # movement noise

for i in range(N):
    kf.predict(movement) # predict state using movement and physical model
    kf.update(zs[i])     # update state using sensor values
    # # save for latter plotting
    state_estimate_x.append(kf.x)
    state_variance_P.append(kf.P)
    # plot the filter output and the variance
    plt.scatter(range(N), zs, marker='+', s=64, color='r', label='measurements')
    p1, = plt.plot(state_estimate_x, label='filter')
    plt.legend(loc='best')
    plt.xlim((0,N));plt.ylim((0,30))
    plt.show()
    sleep(1)
    print(i, 'Last voltage is', state_estimate_x[-1])

plt.plot(state_variance_P)
plt.title('Variance')
plt.show()
print('Variance converges to',state_variance_P[-1])


