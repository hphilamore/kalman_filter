import numpy as np
import matplotlib.pyplot as plt

def bar_chart(hallway, p):

    x_axis = list(range(len(hallway)))
    y_axis = p

    plt.bar(x_axis, y_axis)
    # plt.title('title name')
    # plt.xlabel('x_axis name')
    # plt.ylabel('y_axis name')
    plt.show()

# Scale probability that robot is at this location by factor representing likelihood that sensor is correct

def predict(pos, move, p_correct, p_under, p_over):
    n = len(pos)
    result = np.array(pos, dtype=float)
    for i in range(n):
        result[i] = \
        pos[(i-move) % n] * p_correct + \
        pos[(i-move-1) % n] * p_over + \
        pos[(i-move+1) % n] * p_under
    return result

def normalize(p):
    s = sum(p)
    for i in range (len(p)):
        p[i] = p[i] / s
def update(pos, measure, p_hit, p_miss):
    """
    :param pos: The current probability distribution
    :param measure: The value meaured by the sensor
    :param p_hit: Scale factor showing likelihood that the value is correct (vs incorrect)
    :param p_miss: Scale factor showing likelihood that the value is incorrect (vs correct)
    :return: Probability distribution modified using updated sensor values and scale factors representing
    sensor accuracy
    """

    q = np.array(pos, dtype=float)
    # At each lcoation in the hallyway
    for i in range(len(hallway)):

        # Where the location matches the current sensor data
        if hallway[i] == measure:
            # Scale probability that robot is at this location by factor representing likelihood sensor is correct
            # (i.e. sensor is reading true information at current location)
            q[i] = pos[i] * p_hit

        # Where the location is different to the current sensor data
        else:
            # Scale probability that robot is at this location by factor representing likelihood sensor is incorrect
            # (i.e. sensor is reading false information at current location)
            q[i] = pos[i] * p_miss

    normalize(q)
    return q

# p = np.array([0,0,0,1,0,0,0,0])
hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
sensor = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0])

#p = np.array([0, 0, .4, .6, 0, 0, 0, 0])
#p = hallway * 0.3

#  Initial probability distribution no knowledge and assign equal probability to all positions
# p = np.array([.1]*10)
p = np.array([1/len(hallway)]*len(hallway))

# Sensor is correct at each hallway location
# for location in list(hallway):

# Sensor readings at each hallyway location are given as array
for location in list(sensor):
    # Move 1 place, correct 80%, overshoot 10%, undershoot 10%
    p = predict(p, 1, .8, .1, .1)
    print('predict: ', p)
    # Update probability distribution using sensor readings
    # (sensor readings are 3 x more likely to be correct than incorrect)
    p = update(p, location, 3, 1)
    print('update: ', p)
    print()
    bar_chart(hallway, p)
