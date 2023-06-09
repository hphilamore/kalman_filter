import numpy as np

# Author: Addison Sears-Collins
# https://automaticaddison.com
# Description: A state space model for a differential drive mobile robot


# A = i x i matrix -> i = number of states
# Expresses how the state of robot [e.g. x,y,yaw] changes
# from t-1 to t when no control command is executed.
# For this case, A is the identity matrix.
A_t_minus_1 = np.array([[1.0, 0, 0],
                        [0, 1.0, 0],
                        [0, 0, 1.0]])


# Initial estimated state vector at time t-1
# Global reference frame
# [x_t_minus_1, y_t_minus_1, yaw_t_minus_1]
# [meters, meters, radians]
state_estimate_t_minus_1 = np.array([0.0, 0.0, 0.0])

# Control input vector at time t-1 in global reference frame
# [v, yaw_rate]
# [meters/second, radians/second]
control_vector_t_minus_1 = np.array([4.5, 0.05])

# Noise applied to the forward kinematics
# (calculation of the estimated state at time t using state transition model)
# i length vector -> i = number of states
process_noise_v_t_minus_1 = np.array([0.01,0.01,0.003])

yaw_angle = 0.0 # radians
delta_t = 1.0 # seconds

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

# Example value of estimated state vector at time t
# Global reference frame
# [x_t, y_t, yaw_t]
# [meters, meters, radians]
state_estimate_t_ = np.array([5.2,2.8,1.5708])

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




def main():
    state_estimate_t = (A_t_minus_1 @ (state_estimate_t_minus_1) +
                        getB(yaw_angle, delta_t) @ (control_vector_t_minus_1) +
                        process_noise_v_t_minus_1)
    print(f'State at time t-1: {state_estimate_t_minus_1}')
    print(f'Control input at time t-1: {control_vector_t_minus_1}')
    print(f'State at time t: {state_estimate_t}')  # State after delta_t seconds


    estimated_sensor_observation_y_t = H_t @ state_estimate_t + sensor_noise_w_t
    # print(f'State at time t: {state_estimate_t}')
    print(f'Estimated sensor observations at time t: {estimated_sensor_observation_y_t}')



main()