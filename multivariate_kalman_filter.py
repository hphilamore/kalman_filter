from scipy.stats import multivariate_normal
import numpy as np
from matplotlib import pyplot as plt

"""
Find probability at x,y position of interest 
"""
# position of interest
x = [2.5, 7.3]
x = [2, 7]

# mean position
mu = [2.0, 7.0]

# covariance matrix (covariance xy = 0)
P = [[8.,  0.],
     [0., 10.]]

probability_at_x = multivariate_normal.pdf(x, mu, P)
print(f' probability of being at {x} is {round(probability_at_x * 100, 3)} %')

###########################################

"""
Generate 2D probability distribution 
"""

# range of x and y
x, y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.dstack((x, y))

# mean and covariance
mu = [0.5, -0.2]
P = [[2.0, 0.3],
     [0.3, 0.5]]

# P = [[2.0, 0.0],
#      [0.0, 2.0]]

# P = [[2.0, 1.2],
#      [1.2, 2.0]]

# fix the mean and covariance parameters
# return a “frozen” multivariate normal random variable
rv = multivariate_normal(mu, P)

# plot the distribution
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contourf(x, y, rv.pdf(pos))
plt.show()




