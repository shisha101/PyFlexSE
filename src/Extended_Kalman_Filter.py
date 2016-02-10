import numpy as np
from casadi import *
from robot_model_1 import RobotModel3D  # this is just for trial
rob = RobotModel3D(1.5, 0.33)  # global for coding help

class KF:
    """
    Base class for the Kalman filters containing the class variables which will be inherited by all other
    KF implementations
    """
    # TODO: check if there is an advantage to using Dmatrix instead of np.eye
    def __init__(self, input_system, X0=None, Qin=None, Rin=None, P0=None):
        self.system = input_system

        # system dimensions
        self.ns = self.system.state[0]  # number of states
        self.no = self.system.outputs[0]  # number of outputs
        self.ni = self.system.inputs[0]  # number of inputs

        # system (numeric)
        self.A = None
        self.B = None
        self.C = None
        self.L = None
        self.t = 0.0
        if Qin is None:
            print "WARNING: The Kalman filter's system covariance has not been initialized, system will use I"
            self.Q = np.eye(self.ns)
        else:
            self.Q = Qin  # sys cov
        if Rin is None:
            print "WARNING: The Kalman filter's measurement covariance has not been initialized, system will use I"
            self.R = np.eye(self.no)
        else:
            self.R = Rin  # measurement cov

        # additional
        self.A_t = None   # A transpose to reduce computational costs
        self.L_t = None   # A transpose to reduce computational costs
        self.X = None

        # outputs
        if X0 is None:
            print "WARNING: The Kalman filter's initial state has not been initialized"
        self.X_k_1_c = X0  # the most recent correction value
        self.X_k_1_p = X0  # this will be the most updated estimate (after correction or duing prediction) correction
        # will over write this value to keep it updated for the next prediction
        if P0 is None:
            print "WARNING: The Kalman filter's initial covariance has not been initialized, system will use I"
            self.P_k_1_c = np.eye(self.ns)  # the most recent correction value (only values of corrected covariances)
            self.P_k_1_p = np.eye(self.ns)  # this will be the most updated estimate (after correction or duing prediction)
            # correction will over write this value to keep it updated for the next prediction
        else:
            self.P_k_1_c = P0  # the most recent correction value (only values of corrected covariances)
            self.P_k_1_p = P0  # this will be the most updated estimate (after correction or duing prediction)
            # correction will over write this value to keep it updated for the next prediction
        self.K = None  # the Kalman filter Gain
        self.I = np.eye(self.ns)  # implement an Identity matrix here instead of creating a new one every iteration

    def mul_3(self, m1, m2, m3):
        # TODO: create unit test for this function
        return mul(m1, mul(m2, m3))


class DiscreteKF(KF):
    """
    implementation of Discreet time Kalman filter
    all notations here are assumed to be in hat as in expected x_dot really means x_dot_hat
    x_dot_k_1_c = x_dot @ time k-1(k_1) corrected(_c)
    x_dot_k_p = x_dot @ time k
    x_dot_k_c = x_dot @ time k corrected(c) with the measurements at time K
    in the case of discreet there is no dot ... in variable defs
    .T =  transpose
    Capitals mean Vectors

    we assume the system has the following form:
    X_k = A_k_1 * X_k_1 + B_k_1 * U_k_1 + L_k_1 * W_k_1
    Note that for the matrix multiplying the gaussian noise vector could also be written as
    L_k_1 * W_k_1 -> L_k_1 * W_k_1 * L_k_1.T which we then name W_k_1 (tilde)

    in essence this can all be written with 1 P and one X vector, but to be able to debug and produce good plots,
    it might be desirable to keep different versions and see how they change over the course of time
    """
    # TODO: maybe evaluate the complete prediction function as an SXFunction to speed
    # up evaluation what remains is then the substitution (evaluation)
    # TODO: add some constant calculations during the initialization of the
    # class to save repetitive calculations of static matrices
    # TODO: add unit-tests for this class
    def __init__(self, input_system, X0=None, Qin=None, Rin=None, P0=None):
        KF.__init__(self, input_system, X0=None,Qin=None, Rin=None, P0=None)

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

    def correct(self, system_output):
        # TODO: continue implementing this function
        # TODO: this function should be able to handle asynchronous updates
        # K_c = P_k_1_p * C.T * (C * P_k_1_p* C.T + R)^-1
        K_c = self.mul_3(self.P_k_1_p, self.C.T, inv(self.mul_3(self.C, self.P_k_1_p,self.C.T) + self.R))
        # X_k_c = x_k_1_p + K_k * (Y_k - C_k * x_k_1_p) where here X_k_1_p is our last estimate,
        # which must be a prediction since we predict before we correct
        X_k_c = self.X_k_1_p + mul(K_c, (system_output - mul(self.C, self.X_k_1_p)))

        # P_k_c = (I - K_k * C_k) * P_k_1 * (I - K_k * C_k).T + K_k * R_k * K_k.T
        M_1 = (self.I - mul(K_c, self.C))  # var to avoid multiple matrix multiplications
        P_k_c = self.mul(M_1, self.P_k_l_p, M_1.T) + self.mul_3(K_c, self.R, self.K.T)  # covariance correction

        self.X_k_1_p = X_k_c  # updating var with corrected value
        self.P_k_1_p = P_k_c  # updating var with corrected value
        self.K = K_c


class HybridEKF(KF):
    """
    The Hybrid EKF deals with systems that are governed by continues dynamics but the outputs are measured discretely
    The Hybrid simulates the model during prediction steps (integrates the systems equations),
    and propagates the estimate covariance also via integration
    The correction step for both the states and the covariance is then done in discrete time

    for the EKF to fully function it needs, the systems current input, the last estimated state,
    the system cov, the sensor cov, the most recent output during the correction phase, and the system equations
    """
    # TODO: re-evaluate integrator if time step for integration changes
    # TODO: function for matrix re-evaluation
    # TODO: function for X_dot
    # TODO: function for P_dot
    # TODO: initial state (where should it come from, system ? or KF) -> KF
    # TODO: join P_dot to X_dot Integrator to create a single integrator
    # TODO: initial P_dot if not supplied
    """
        fin = controldaeIn(t=self.t_sym, x=self.state, p=self.p_sym, u=self.inputs)
        X_dot_function = SXFunction('f', fin, daeOut(ode=self.X_dot))
        I_options = {}
        I_options["t0"] = 0
        I_options["tf"] = 50
        I_options["integrator"] = "cvodes"
        csim = ControlSimulator("f", X_dot_function, I_options)
    """
    def __init__(self, input_system, X0=None, Qin=None, Rin=None, P0=None):
        KF.__init__(self, input_system, X0=None, Qin=None, Rin=None, P0=None)

        self.integ_ts = 0.2  # time step used for integration
        self.solver_integ = "cvodes"

        # symbolic variables
        self.P_sym = SX.sym("p", self.ns, self.ns)  # nxn symbolic matrix
        self.t_sym = SX.sym("t")  # symbolic time
        self.A_sym = SX.sym("a", self.ns, self.ns)
        # not being used yet
        self.L_sym = None
        self.M_sym = None
        self.C_sym = None

        # functions used for linearization
        if self.system.jac_X_dot_wrt_x is None:  # the jacobian has not been computed for this system yet
            self.system.compute_jac_Xdot()
        self.A_func = self.system.jac_X_dot_wrt_x
        # not being used yet
        self.L_func = None

        if self.system.output_output_SX is None:  # no output equations specified assume Identity
            self.C = np.eye(self.ni)  # HACK
        else:
            self.C_func = self.system.jac_output_wrt_x
            # not being used yet
            self.M_func = None


        # functions
        self.cov_estimate_func = None

        # integrators
        self.system_integrator = None
        self.estimate_cov_integrator = None

    def create_p_dot_function(self):
        P_dot = mul(self.A_sym, self.P_k_1_p) + mul(self.P_k_1_p, self.A_sym.T) + self.Q # TODO: generalize to L * Q * L.T and make it a parameter in the eq below
        f_in = daeIn(x=self.P_sym, p=self.A_sym, t=self.t_sym)  # Q is considered a constant
        f_out = daeOut(ode=P_dot)
        self.cov_estimate_func = SXFunction("estimate_cov_function", f_in, f_out)

    def create_integrators(self):
        # Integrator("name", solver, function, optionsDict)
        sys_integ_opt = {}
        # TODO: change the constant time step to maybe use seconds from start of estimation
        sys_integ_opt["t0"] = 0.0
        sys_integ_opt["tf"] = self.integ_ts
        self.system_integrator = Integrator("state integrator", self.solver_integ, self.system.X_dot_func,
                                            sys_integ_opt)
        self.estimate_cov_integrator = Integrator("state covariance integrator", self.solver_integ,
                                                  self.cov_estimate_func, sys_integ_opt)

    def prediction(self, system_input):
        self.update_prediction_matrices(system_input)  # the A matrix
        # TODO: note the Q matrix is not Q tilde yet assuming that L is I
        # integrate the covariance
        self.estimate_cov_integrator.setInput(self.P_k_1_p, "x0")
        self.estimate_cov_integrator.setInput(self.A, "p")
        self.estimate_cov_integrator.evaluate()
        print self.estimate_cov_integrator.getOutput()
        print "the type of the cov output is %s, the output is %s" %\
        (type(self.estimate_cov_integrator.getOutput()[-1]), self.estimate_cov_integrator.getOutput()[-1])

        # TODO: add the var update

        # integrate the states
        self.system_integrator.setInput(self.X_k_1_p, "x0")
        self.system_integrator.setInput(vertcat(system_input, self.system.p_num), "p")
        self.system_integrator.evaluate()
        print self.system_integrator.getOutput()
        print "the type of the system output is %s, the output is %s" %\
        (type(self.system_integrator.getOutput()[-1]), self.system_integrator.getOutput()[-1])
        # TODO: add the var update

    def correction(self, system_output):
        self.update_correction_matrices()  # the C matrix evaluation
        # TODO: note the R matrix is not R tilde yet assuming that M is I

        # correction gain calculation
        # K_c = P_k_1_p * C.T * (C * P_k_1_p* C.T + R)^-1
        K_c = self.mul_3(self.P_k_1_p, self.C.T, inv(self.mul_3(self.C, self.P_k_1_p,self.C.T) + self.R))

        # state correction
        estimated_output = self.get_estimated_output()
        # X_k_c = x_k_1_p + K_k * (Y_k - h(x_k_1, 0, t_k)) where here X_k_1_p is our last estimate,
        # which must be a prediction since we predict before we correct
        X_k_c = self.X_k_1_p + mul(K_c, (system_output - estimated_output))

        # covariance correction
        # P_k_c = (I - K_k * C_k) * P_k_1 * (I - K_k * C_k).T + K_k * R_k * K_k.T
        M_1 = (self.I - mul(K_c, self.C))  # var to avoid multiple matrix multiplications
        P_k_c = self.mul_3(M_1, self.P_k_1_p, M_1.T) + self.mul_3(K_c, self.R, K_c.T)  # covariance correction

        self.P_k_1_c = P_k_c
        self.P_k_1_p = P_k_c
        self.X_k_1_c = X_k_c
        self.X_k_1_p = X_k_c
        self.K = K_c

    def update_prediction_matrices(self, system_input):
        # the A matrix must be evaluated and the L matrix the L matrix for later
        A_matrix_linearized = self.A_func({"x": self.X_k_1_p,
                                           "p": vertcat([system_input, self.system.p_num]),
                                           "t": self.t})
        self.A = A_matrix_linearized

    def update_correction_matrices(self):
        # the C matrix amd the M matrix the M matrix for later
        pass

    def get_estimated_output(self):
        pass

    def update_EKF(self):
        pass


class UKF(KF):
    def __init__(self, input_system):
        KF.__init__(self, input_system)

    def pick_sigma_points(self):
        # function
        pass
