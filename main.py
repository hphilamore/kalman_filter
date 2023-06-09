# Author: Addison Sears-Collins
# https://automaticaddison.com
# Description: A state space model for a differential drive mobile robot
# Description: An observation model for a differential drive mobile robot
# Description: Extended Kalman Filter example (two-wheeled mobile robot)
import numpy as np

# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=3,suppress=True)

# # Time step
# dt = 1.0  # seconds

# A = i x i matrix -> i = number of states
# Expresses how the state of robot [e.g. x,y,yaw] changes
# from t-1 to t when no control command is executed.
# For this case, A is the identity matrix.
A = np.array([[1.0, 0, 0],
              [0, 1.0, 0],
              [0, 0, 1.0]])


# # Initial estimated state vector at time t-1
# # Global reference frame
# # [x_t_minus_1, y_t_minus_1, yaw_t_minus_1]
# # [meters, meters, radians]
# x = np.array([0.0, 0.0, 0.0])

# # Control input vector at time t-1 in global reference frame
# # [v, yaw_rate]
# # [meters/second, radians/second]
# u = np.array([4.5, 0.05])

# Process noise applied to the forward kinematics
# (calculation of the estimated state at time t using state transition model)
# i length vector -> i = number of states
v = np.array([0.01,0.01,0.003])

yaw_angle = 0.0 # radians


# Measurement matrix H
# Converts predicted state estimate, time t, to predicted sensor measurements, time t.
# For this case, H will be the identity matrix (estimated state maps directly to state
# measurements from robot odometry data [x, y, yaw])
# k x i matrix -> k = number of sensor measurements, i = number of states.
H = np.array([    [1.0,  0,    0],
                    [0,    1.0,  0],
                    [0,    0,    1.0]])

# Sensor noise.
# k length vector -> k = number of sensor measurements
w = np.array([0.07,0.07,0.04])

# # Example value of estimated state vector at time t
# # Global reference frame
# # [x_t, y_t, yaw_t]
# # [meters, meters, radians]
# x_ = np.array([5.2,2.8,1.5708])

# # State covariance matrix P
# # i x i matrix -> i = number of states
# # An estimate of the accuracy of the state estimate at time t, made using the
# # state transition matrix.
# # Initialise with guessed values.
# P = np.array([[0.1, 0, 0],
#                         [0, 0.1, 0],
#                         [0, 0, 0.1]])

# State model noise covariance matrix Q_k
# When Q is large, Kalman Filter tracks large changes in sensor measurements more closely.
# i x i matrix -> i = number of states.
Q = np.array([[1.0,   0,   0],
                [  0, 1.0,   0],
                [  0,   0, 1.0]])

# Sensor measurement noise covariance matrix R_k
# k x k matrix -> k = number of sensor measurements
# R -> 0 as sensor accuracy -> 100%
R = np.array([[1.0, 0, 0],
                [0, 1.0, 0],
                [0, 0, 1.0]])

# Sensor noise
# k length vector -> k = number of sensor measurements
# w = np.array([0.07, 0.07, 0.04])



def getB(yaw, dt):
  """
  Calculates and returns the B
  ixj matrix -> i states x j control inputs

  Control inputs: linear and angular velocity [v, yaw_rate]

  Expresses how the state of the robot [e.g. x,y,yaw] changes
  from t-1 to t due to the control commands (inputs).
  :param yaw: The yaw (rotation angle around the z axis) in rad
  :param dt: The change in time from time step t-1 to t in sec
    """
  B = np.array([[np.cos(yaw)*dt, 0],
                [np.sin(yaw)*dt, 0],
                [0, dt]])
  return B


def ekf(z,x,u,P,dt):

    """
    Extended Kalman Filter. Fuses noisy sensor measurement to
    create an optimal estimate of the state of the robotic system.

    INPUT
        :param z The observation from the Odometry
            3x1 NumPy Array [x,y,yaw] in the global reference frame
            in [meters,meters,radians].
        :param x The state estimate at time k-1
            3x1 NumPy Array [x,y,yaw] in the global reference frame
            in [meters,meters,radians].
        :param u The control vector applied at time k-1
            3x1 NumPy Array [v,v,yaw rate] in the global reference frame
            in [meters per second,meters per second,radians per second].
        :param P The state covariance matrix estimate at time k-1
            3x3 NumPy Array
        :param dt Time interval in seconds

    OUTPUT
        :return state_estimate_k near-optimal state estimate at time k
            3x1 NumPy Array ---> [meters,meters,radians]
        :return P_k state covariance_estimate for time k
            3x3 NumPy Array
    """


    ######################### Predict #############################
    # Predict state estimate at time t based on state
    # estimate at time t-1 and control input applied at time t-1
    x = (A @ x + getB(x[2], dt) @ u + v)
    print(f'State Estimate Before EKF={x}')

    # Predict state covariance estimate based on previous covariance and some noise
    P = A @ P @ A.T + Q

    ################### Update (Correct) ##########################
    # Calculate the difference between the actual sensor measurements at time t minus
    # sensor measurements predicted by the measurement model for time t.
    y = z - (H @ x + w)
    print(f'Observation={z}')

    # Calculate the measurement residual covariance
    S = H @ P @ H.T + R

    # Calculate the near-optimal Kalman gain
    # (We use pseudoinverse since some of the matrices might be non-square or singular)
    K = P @ H.T @ np.linalg.pinv(S)

    # Calculate an updated state estimate for time t
    x = x + (K @ y)

    # Update the state covariance estimate for time t
    P = P - (K @ H @ P)

    # Print the best (near-optimal) estimate of the current state of the robot
    print(f'State Estimate After EKF={x}')

    # Return the updated state and covariance estimates
    return x, P


def main():
    # Time step
    dt = 1.0  # seconds

    # Start at time t = 1
    t = 1

    # Initial estimated state vector at time t-1
    # Global reference frame
    # [x_t_minus_1, y_t_minus_1, yaw_t_minus_1]
    # [meters, meters, radians]
    x = np.array([0.0, 0.0, 0.0])

    # Control input vector at time t-1 in global reference frame
    # [v, yaw_rate]
    # [meters/second, radians/second]
    u = np.array([4.5, 0.05])

    # State covariance matrix P
    # i x i matrix -> i = number of states
    # An estimate of the accuracy of the state estimate at time t, made using the
    # state transition matrix.
    # Initialise with guessed values.
    P = np.array([[0.1, 0, 0],
                  [0, 0.1, 0],
                  [0, 0, 0.1]])



    # A list of sensor observations at successive timesteps
    # Each list within z is a 3-element observation vector.
    z_all = np.array([[4.721,0.143,0.006], # k=1
                    [9.353,0.284,0.007], # k=2
                    [14.773,0.422,0.009],# k=3
                    [18.246,0.555,0.011], # k=4
                    [22.609,0.715,0.012]])# k=5


    for t, z in enumerate(z_all, start=1):
        # Print the current timestep
        print(f'Timestep k={t}')


        x, P = ekf(z, x, u, P, dt)

        print()


main()