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
A_t_minus_1 = np.array([[1.0, 0, 0],
                        [0, 1.0, 0],
                        [0, 0, 1.0]])


# # Initial estimated state vector at time t-1
# # Global reference frame
# # [x_t_minus_1, y_t_minus_1, yaw_t_minus_1]
# # [meters, meters, radians]
# state_estimate_t_minus_1 = np.array([0.0, 0.0, 0.0])

# # Control input vector at time t-1 in global reference frame
# # [v, yaw_rate]
# # [meters/second, radians/second]
# control_vector_t_minus_1 = np.array([4.5, 0.05])

# Noise applied to the forward kinematics
# (calculation of the estimated state at time t using state transition model)
# i length vector -> i = number of states
process_noise_v_t_minus_1 = np.array([0.01,0.01,0.003])

yaw_angle = 0.0 # radians


# Measurement matrix H_t
# Converts predicted state estimate, time t, to predicted sensor measurements, time t.
# For this case, H will be the identity matrix (estimated state maps directly to state
# measurements from robot odometry data [x, y, yaw])
# k x i matrix -> k = number of sensor measurements, i = number of states.
H_t = np.array([    [1.0,  0,    0],
                    [0,    1.0,  0],
                    [0,    0,    1.0]])

# Sensor noise.
# k length vector -> k = number of sensor measurements
sensor_noise_w_t = np.array([0.07,0.07,0.04])

# # Example value of estimated state vector at time t
# # Global reference frame
# # [x_t, y_t, yaw_t]
# # [meters, meters, radians]
# state_estimate_t_ = np.array([5.2,2.8,1.5708])

# # State covariance matrix P_t_minus_1
# # i x i matrix -> i = number of states
# # An estimate of the accuracy of the state estimate at time t, made using the
# # state transition matrix.
# # Initialise with guessed values.
# P_t_minus_1 = np.array([[0.1, 0, 0],
#                         [0, 0.1, 0],
#                         [0, 0, 0.1]])

# State model noise covariance matrix Q_k
# When Q is large, Kalman Filter tracks large changes in sensor measurements more closely.
# i x i matrix -> i = number of states.
Q_t = np.array([[1.0,   0,   0],
                [  0, 1.0,   0],
                [  0,   0, 1.0]])

# Sensor measurement noise covariance matrix R_k
# k x k matrix -> k = number of sensor measurements
# R -> 0 as sensor accuracy -> 100%
R_t = np.array([[1.0, 0, 0],
                [0, 1.0, 0],
                [0, 0, 1.0]])

# Sensor noise
# k length vector -> k = number of sensor measurements
sensor_noise_w_t = np.array([0.07, 0.07, 0.04])



def getB(yaw,dt):
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


def ekf(z_t_observation_vector,
        state_estimate_t_minus_1,
        control_vector_t_minus_1,
        P_t_minus_1,
        dt):
# def ekf(z_t_observation_vector):
    """
    Extended Kalman Filter. Fuses noisy sensor measurement to
    create an optimal estimate of the state of the robotic system.

    INPUT
        :param z_k_observation_vector The observation from the Odometry
            3x1 NumPy Array [x,y,yaw] in the global reference frame
            in [meters,meters,radians].
        :param state_estimate_k_minus_1 The state estimate at time k-1
            3x1 NumPy Array [x,y,yaw] in the global reference frame
            in [meters,meters,radians].
        :param control_vector_k_minus_1 The control vector applied at time k-1
            3x1 NumPy Array [v,v,yaw rate] in the global reference frame
            in [meters per second,meters per second,radians per second].
        :param P_k_minus_1 The state covariance matrix estimate at time k-1
            3x3 NumPy Array
        :param dk Time interval in seconds

    OUTPUT
        :return state_estimate_k near-optimal state estimate at time k
            3x1 NumPy Array ---> [meters,meters,radians]
        :return P_k state covariance_estimate for time k
            3x3 NumPy Array
    """


    ######################### Predict #############################
    # Predict state estimate at time t based on state
    # estimate at time t-1 and control input applied at time t-1
    state_estimate_t = (A_t_minus_1 @ state_estimate_t_minus_1 +
                        getB(state_estimate_t_minus_1[2], dt) @ control_vector_t_minus_1 +
                        process_noise_v_t_minus_1)
    print(f'State Estimate Before EKF={state_estimate_t}')

    # Predict state covariance estimate based on previous covariance and some noise
    P_t = A_t_minus_1 @ P_t_minus_1 @ A_t_minus_1.T + Q_t

    ################### Update (Correct) ##########################
    # Calculate the difference between the actual sensor measurements at time t minus
    # sensor measurements predicted by the measurement model for time t.
    measurement_residual_y_t = (z_t_observation_vector -
                                (H_t @ state_estimate_t + sensor_noise_w_t)
                                )

    print(f'Observation={z_t_observation_vector}')

    # Calculate the measurement residual covariance
    S_t = H_t @ P_t @ H_t.T + R_t

    # Calculate the near-optimal Kalman gain
    # (We use pseudoinverse since some of the matrices might be non-square or singular)
    K_t = P_t @ H_t.T @ np.linalg.pinv(S_t)

    # Calculate an updated state estimate for time t
    state_estimate_t = state_estimate_t + (K_t @ measurement_residual_y_t)

    # Update the state covariance estimate for time t
    P_t = P_t - (K_t @ H_t @ P_t)

    # Print the best (near-optimal) estimate of the current state of the robot
    print(f'State Estimate After EKF={state_estimate_t}')

    # Return the updated state and covariance estimates
    return state_estimate_t, P_t


def main():
    # Time step
    dt = 1.0  # seconds

    # Start at time t = 1
    t = 1

    # Initial estimated state vector at time t-1
    # Global reference frame
    # [x_t_minus_1, y_t_minus_1, yaw_t_minus_1]
    # [meters, meters, radians]
    state_estimate_t_minus_1 = np.array([0.0, 0.0, 0.0])

    # Control input vector at time t-1 in global reference frame
    # [v, yaw_rate]
    # [meters/second, radians/second]
    control_vector_t_minus_1 = np.array([4.5, 0.05])

    # State covariance matrix P_t_minus_1
    # i x i matrix -> i = number of states
    # An estimate of the accuracy of the state estimate at time t, made using the
    # state transition matrix.
    # Initialise with guessed values.
    P_t_minus_1 = np.array([[0.1, 0, 0],
                            [0, 0.1, 0],
                            [0, 0, 0.1]])



    # A list of sensor observations at successive timesteps
    # Each list within z_t is a 3-element observation vector.
    z_t = np.array([[4.721,0.143,0.006], # k=1
                    [9.353,0.284,0.007], # k=2
                    [14.773,0.422,0.009],# k=3
                    [18.246,0.555,0.011], # k=4
                    [22.609,0.715,0.012]])# k=5


    # state_estimate_t = (A_t_minus_1 @ (state_estimate_t_minus_1) +
    #                     getB(yaw_angle, dt) @ (control_vector_t_minus_1) +
    #                     process_noise_v_t_minus_1)
    # print(f'State at time t-1: {state_estimate_t_minus_1}')
    # print(f'Control input at time t-1: {control_vector_t_minus_1}')
    # print(f'State at time t: {state_estimate_t}')  # State after delta_t seconds
    #
    #
    # estimated_sensor_observation_y_t = H_t @ state_estimate_t + sensor_noise_w_t
    # # print(f'State at time t: {state_estimate_t}')
    # print(f'Estimated sensor observations at time t: {estimated_sensor_observation_y_t}')

    for t, obs_vector_z_t in enumerate(z_t, start=1):
        # Print the current timestep
        print(f'Timestep k={t}')


        optimal_state_estimate_t, covariance_estimate_t = ekf(obs_vector_z_t,
                                                              state_estimate_t_minus_1,
                                                              control_vector_t_minus_1,
                                                              P_t_minus_1,
                                                              dt)

        # optimal_state_estimate_t, covariance_estimate_t = ekf(obs_vector_z_t)

        # Get ready for the next timestep by updating the variable values
        state_estimate_t_minus_1 = optimal_state_estimate_t
        P_t_minus_1 = covariance_estimate_t

        print()


main()