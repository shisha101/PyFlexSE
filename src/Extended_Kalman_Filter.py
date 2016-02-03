import numpy as np
from casadi import *
from robot_model_1 import SystemModel, RobotModel3D, RobotModel2D  # this is just for trial
rob = RobotModel3D(1.5, 0.33)  # global for coding help

class KF:
    """
    implementation if Discreet time Kalman filter
    all notations here are assumed to be in hat as in expected x_dot really means x_dot_hat
    x_dot_k_1_c = x_dot @ time k-1(k_1) corrected(_c)
    x_dot_k_p = x_dot @ time k
    x_dot_k_c = x_dot @ time k corrected(c) with the measurements at time K
    in the case of discreet there is no dot ... in variable defs
    .T =  transpose
    Capitals mean Vectors

    we assume the system has the following form:
    X_k = A_k_1 * X_k_1 + B_k_1 * U_k_1 + L_k_1 * W_k_1
    Note that for the matrix multiplying the gausian noise vector could also be written as
    L_k_1 * W_k_1 -> L_k_1 * W_k_1 * L_k_1.T which we then name W_k_1 (tilde)

    in essence this can all be written with 1 P and one X vector, but to be able to debug and produce good plots,
    it might be desirable to keep different versions and see how they change over the course of time
    """
    # TODO: maybe evaluate the complete prediction function as an SXFunction to speed
    # up evaluation what remains is then the substitution (evaluation)
    # TODO: add some constant calculations during the initalization of the
    # class to save repetitive calculations of static matrices
    def __init__(self, input_system):
        self.system = input_system
        # system
        self.A = None
        self.B = None
        self.C = None
        self.L = None
        self.Q = None

        # additional
        self.A_t = None   # A transpose to reduce computational costs
        self.L_t = None   # A transpose to reduce computational costs
        self.X = None

        # outputs
        self.X_k_1_c = None # the most recent correction value
        self.X_k_1_p = None# this will be the most uptodate estimate (after correction or duing prediction) correction
        # will over write this value to keep it uptodate for the next prediction
        self.P_k_1_c = None # the most recent correction value
        self.P_k_1_p = None# this will be the most uptodate estimate (after correction or duing prediction) correction
        # will over write this value to keep it uptodate for the next prediction
        self.K = None  # the Kalman filter Gain
        self.I = None  # implement an Identity matrix here instead of creating a new one every iteration

    def predict(self, system_input):
        # all in hat,x_hat (expected)
        # note that the X_k_p and P_k_p are updated and over written

        # X_k_p = A * X_k_1_c + B_K_1 * U_k_1
        X_k_p = mul(self.A, self.X_k_1_p) + mul(self.B, system_input)
        self.X_k_1_p = X_k_p  # the best current estimate based on prediction (no correction information yet)

        # this operation could technically be postponed to the correction step where it is needed but it is usfull to
        # have for inspection purposes on how the cov of our extimate changes with prediction
        # P_k_p = A * P_k_1_c * A.T + Q_k_1
        P_k_p = mul(self.A, mul(self.P_k_1_p, self.A_t)) + mul(self.L, mul(self.Q, self.L_t))
        self.P_k_1_p = P_k_p

    def correct(self):
        # TODO: continue implementing this function
        (self.I - mul(self.K, self.C))


        self.X_k_1_p = X_k_c  # updating var with corrected value
        self.P_k_1_p = P_k_c  # updating var with corrected value

class EKF(KF):
    def __init__(self, input_system):
        KF.__init__(self, input_system)
        # calculate derevatives
        # do other things

    def update_EKF(self):
        x = 0

class UKF(KF):
    def __init__(self, input_system):
        KF.__init__(self, input_system)

    def pick_sigma_points(self):
        # function
        x = 0
