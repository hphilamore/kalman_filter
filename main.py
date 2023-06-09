import numpy as np

# Author: Addison Sears-Collins
# https://automaticaddison.com
# Description: A state space model for a differential drive mobile robot


# A = n x n matrix -> n = number of states
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
# n length vector -> n = number of states
process_noise_v_t_minus_1 = np.array([0.01,0.01,0.003])

yaw_angle = 0.0 # radians
delta_t = 1.0 # seconds

def getB(yaw,dt):
  """
  Calculates and returns the B
  nxm matix -> n states x m control inputs

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


main()