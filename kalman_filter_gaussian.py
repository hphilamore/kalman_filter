from __future__ import print_function, division
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy.random as random
import numpy as np
import math

class DogSensor():
    def __init__(self, x0=0, velocity=1, noise=0.0):
        """
        x0 : initial position ground truth
        velocity : (+ve values = move right, -ve values = move left)
        noise : scaling factor for noise, 0 = no noise
        """
        self.x = x0
        self.velocity = velocity
        self.noise = np.sqrt(noise)
    def sense(self):
        self.x = self.x + self.velocity
        return self.x + random.randn() * self.noise


for i in range(20):
    print('{: 5.4f}'.format(random.randn()),end='\t')
    if (i+1) % 5 == 0:
        print('')

# dog = DogSensor (noise=0.0)
# xs = []
# for i in range(10):
#     x = dog.sense()
#     xs.append(x)
#     print("%.4f" % x, end=' ')
#
# plt.plot(xs, label='dog position')
# plt.legend(loc='best')
# plt.show()



def test_sensor(noise_scale):
    dog = DogSensor(noise=noise_scale)
    xs = []
    for i in range(100):
        x = dog.sense()
        xs.append(x)
    plt.plot(xs, label='sensor')
    plt.plot([0,99],[1,100], 'r--', label='actual')
    plt.xlabel('time')
    plt.ylabel('pos')
    plt.ylim([0,100])
    plt.title('noise = ' + str(noise_scale))
    plt.legend(loc='best')
    plt.show()

def gaussian(x, mean, var):
    """returns normal distribution for x given a
    gaussian with the specified mean and variance.
    """
    return np.exp((-0.5 * (x - mean) ** 2) / var) / np.sqrt(2 * np.pi * var)

def plot_gaussian(x_range, mean, var):
    y_arr = []
    for x in x_range:
        y = gaussian(x, mean, var)
        y_arr.append(y)
    plt.plot(x_range, y_arr)
    plt.show()

def multiply_gaussians(mu1, var1, mu2, var2):
    mean = (var1*mu2 + var2*mu1) / (var1+var2)
    variance = 1 / (1/var1 + 1/var2)
    return (mean, variance)

def update(mean_pos, variance_pos, mean_measurement, variance_measurement):
    return multiply_gaussians(mean_pos, variance_pos, mean_measurement, variance_measurement)

def predict(mean_pos, variance_pos, mean_movement, variance_movement):
    return (mean_pos + mean_movement, variance_pos + variance_movement)


# test_sensor(4.0)

# Dog moving with velocity = 1
# dog = DogSensor(23, 1, 5)
# # dog = DogSensor(noise=0.4)
# xs = range(100)
# ys = []
# for i in xs:
#     ys.append(dog.sense())
# plt.plot(xs,ys, label='dog position')
# plt.legend(loc='best')
# plt.show()

# dog standing still with incrrect initial belief corrected by series of sensor updates
# dog = DogSensor(velocity=0, noise=1)
# pos,s = 2, 5
# for i in range(20):
#     pos,s = update(pos, s, dog.sense(), 5)
#     plot_gaussian(range(-5, 5), pos, s)
#     print('time:', i, '\tposition =', "%.3f" % pos, '\tvariance =', "%.3f" % s)


movement = 0#1
movement_error = 2#30#2
sensor_error = 30#4.5
pos = (1000,500)#(0, 50) # initial belief = gaussian N(mean, variance) = N(0,50)
dog = DogSensor(x0=0, velocity=movement, noise=sensor_error)
zs = [] # array to store series of sensor measurements
ps = [] # array to store series of positions
vs = [] # array to store series of position variance




for i in range(100):
    # predict new pos, given current belief
    pos = predict(mean_pos = pos[0],
                  variance_pos = pos[1],
                  mean_movement = movement,
                  variance_movement = movement_error)
    print('PREDICT: {: 10.4f} {: 10.4f}'.format(pos[0], pos[1]),end='\t')

    # take new sensor measurement
    Z = dog.sense()

    # # alternative sensor measurement, nonlinear function independent of movement
    # Z = math.sin(i / 3.) * 2 + random.randn()*1.2

    zs.append(Z)      # store sensor measurement
    variance_pos = pos[1]
    vs.append(variance_pos) # store position variance

    # update position
    pos = update(mean_pos = pos[0],
                 variance_pos = pos[1],
                 mean_measurement = Z,
                 variance_measurement = sensor_error)
    ps.append(pos[0]) # store updated position

    # print new belief
    print('UPDATE: {: 10.4f} {: 10.4f}'.format(pos[0], pos[1]))

plt.plot(ps, label='filter')
plt.plot(zs, c='r', linestyle='dashed', label='measurement')
plt.legend(loc='best')
plt.show()

plt.plot(vs)
plt.title('Variance')
plt.show()
print ([float("%0.4f" % v) for v in vs])

