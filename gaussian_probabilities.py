import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def gaussian(x, mean, var):
    """returns normal distribution for x given a
    gaussian with the specified mean and variance.
    """
    return np.exp((-0.5*(x-mean)**2)/var) / np.sqrt(2*np.pi*var)

for x in range(12, 32):
    f = gaussian(x, 22, 4)
    plt.plot(x, f, 'o')
plt.show()

print(norm(2,3).pdf(1.5))
print(gaussian(x=1.5, mean=2, var=3*3))

#  Create a ‘frozen’ distribution with a mean of 2 and a standard deviation of 3
n23 = norm(2,3)
#  Get the probability density of various values,
print ('probability density of 1.5 is %.4f' % n23.pdf(1.5))
print ('probability density of 2.5 is also %.4f' % n23.pdf(2.5))
print ('whereas probability density of 2 is %.4f' % n23.pdf(2))

# Generate n samples from distribution
n = 15
print(n23.rvs(size=n))

# Get the cumulative distribution function (CDF)
# Probability that a randomly drawn value from the distribution is <= m
m = 1
print (n23.cdf (m))

m = 2
print (n23.cdf (m)) # 50% of samples are less than the mean

# Various properties of the distribution:
print('variance is', n23.var())
print('standard deviation is', n23.std())
print('mean is', n23.mean())