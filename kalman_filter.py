from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy.random as random
import numpy as np

class DogSensor():
    def __init__(self, x0=0, velocity=1, noise=0.0):
        """ x0 - initial position
        velocity - (+=right, -=left)
        noise - scaling factor for noise, 0== no noise
        """
        self.x = x0
        self.velocity = velocity
        self.noise = np.sqrt(noise)
    def sense(self):
        self.x = self.x + self.velocity
        return self.x + random.randn() * self.noise