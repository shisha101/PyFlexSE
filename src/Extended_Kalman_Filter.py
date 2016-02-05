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
    # TODO: add some constant calculations during the initialization of the
    # class to save repetitive calculations of static matrices
    def __init__(self, input_system):
        self.system = input_system
        # system
        self.A = None
        self.B = None
        self.C = None
        self.L = None
        self.Q = None  # sys cov
        self.R = None  # measurement cov

        # additional
        self.A_t = None   # A transpose to reduce computational costs
        self.L_t = None   # A transpose to reduce computational costs
        self.X = None

        # outputs
        self.X_k_1_c = None # the most recent correction value
        self.X_k_1_p = None# this will be the most uptodate estimate (after correction or duing prediction) correction
        # will over write this value to keep it uptodate for the next prediction
        self.P_k_1_c = None  # the most recent correction value (only values of corrected covariances)
        self.P_k_1_p = None  # this will be the most uptodate estimate (after correction or duing prediction) correction
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
        P_k_p = self.mul_3(self.A, self.P_k_1_p, self.A_t) + self.mul_3(self.L, self.Q, self.L_t)
        self.P_k_1_p = P_k_p

    def correct(self):
        # TODO: continue implementing this function
        # TODO: this function should be able to handle asynchronous updates
        # K_c = P_k_1_p * C.T * (C * P_k_1_p* C.T + R)^-1
        K_c = mul_3(self.P_k_1_p, self.C.T, inv(mul_3(self.C, self.P_k_1_p,) + ))

        M_1 = (self.I - mul(self.K, self.C))  # var to avoid multiple matrix multiplications
        P_k_c = mul(M_1, mul(self.P_k_l_p, M_1.T)) + mul(self.K, mul(self.R * self.K.T))  # covariance correction


        self.X_k_1_p = X_k_c  # updating var with corrected value
        self.P_k_1_p = P_k_c  # updating var with corrected value

    def mul_3(self, m1, m2, m3):
        return mul(m1, mul(m2, m3))

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
