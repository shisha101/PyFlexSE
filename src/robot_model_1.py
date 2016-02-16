import numpy as np
from casadi import *


class SystemModel:
    def __init__(self):
        """
        for the parameter vector make sure that the inputs go in before the parameters(constants/time varying)
        """
        self.p_num = None  # numeric parameters of the system (numpy array)
        self.p_sym = None  # symbolic parameters of the system (allowed to be changed, from one evaluation to the next)
        self.inputs = None  # symbolic system inputs
        self.outputs = None  # the expression (SX) for the output equation describing the outputs(never None after init)
        self.state = None  # the symbolic vector of system states
        # additions
        self.current_state = []

        self.X_dot = None  # expression (SX)
        self.X_dot_func = None  # CasADi (SXFunciton)
        self.jac_X_dot_wrt_x = None  # Joacobian with respect to states
        self.jac_X_dot_wrt_p = None  # Joacobian with respect to parameters(all (inputs + system parameters))
        self.jac_X_dot_complete = None  # Jacobian with respect to all states and parameters as well as time

        self.C = None  # in the case of linear outputs this self.C is set and output_func stays null and vice versa
        self.output_func = None  # CasADi (SXFunction) for the output
        self.jac_output_wrt_x = None  # jacobian of output function wrt states

        # not being used yet
        # self.system_equations = None  # the concatination of the state transition equations and the output equations
        # self.output_eq = None
        # self.jac_complete = None  # complete system jacobian (wrt states, params(all), time)
        # self.jac_sys_wrt_states = None
        # self.jac_sys_wrt_params = None

        self.sensor_cov = None  # R
        self.system_cov = None  # Q
        self.init_estimate_cov = None  # P0
        self.t_sym = SX.sym("t")

    def get_sys_eq(self):
        if self.X_dot is not None:
            return self.X_dot

    # def compute_sys_eq_jac(self):
    #     if self.system_equations is not None:
    #         self.jac_complete = self.system_equations.fullJacobian()
    #         self.jac_sys_wrt_states = self.system_equations.jacobian(0)
    #         self.jac_sys_wrt_params = self.system_equations.jacobian(1)

    def compute_jac_Xdot(self):
        if self.X_dot is not None and self.X_dot_func is not None:
            # self.jac_X_dot_complete = jacobian(
            #         self.X_dot, vertcat([self.state, self.inputs])) # SX expression wrt X and U
            self.jac_X_dot_complete = self.X_dot_func.fullJacobian()
            self.jac_X_dot_wrt_x = self.X_dot_func.jacobian("x")  # (0)
            self.jac_X_dot_wrt_p = self.X_dot_func.jacobian("p")  # (2) # x, z, p, t (the order of vars in the ode)

    def compute_jac_output_func(self):
        """
        This function has not been tested
        """
        if self.outputs is not None and self.output_func is not None:
            jacobian = self.output_func.jacobian("x")
            self.jac_output_wrt_x = jacobian  # (0)
        else:
            print "output function or output expression has not been set, No jacobian calculation possible." \
                  "does the system have a linear output function ?"

    # getter functions
    def get_p_numeric(self):
        if not self.p_num:
            return self.p_num
        else:
            print "no numeric parameters set"

    def get_p_symbolic(self):
        if self.p_sym is not None:
            return self.p_sym
        else:
            print "no symbolic parameters set"

    def get_inputs(self):
        if self.inputs is not None:
            return self.inputs
        else:
            print "no inputs set"

    def get_outputs(self):
        if self.outputs is not None:
            return self.outputs
        else:
            print "no outputs set"

    def get_states(self):
        if self.state is not None:
            return self.state
        else:
            print "no states set, this system has not been initialized correctly"

    def get_state_transiton_eq(self):
        if self.X_dot is not None:
            return self.X_dot
        else:
            print "no states set, this system has not been initialized correctly"

    def get_current_state(self):
        if not self.current_state:
            return self.current_state
        else:
            print "no current state available"

    # debug functions
    def print_debug_casadi(self):  # addition
        if self.X_dot_func is not None:
            print "the number of input(s) %s output(s) %s " % (self.X_dot_func.nIn(), self.X_dot_func.nOut())
            print "the input(s) : %s " % (self.X_dot_func.symbolicInput())
            print "the output(s): %s " % (self.X_dot_func.symbolicOutput())
            print "the system input p is %s " % (self.X_dot_func.getInput("p"))
            print "the jacobian wrt to x is %s" % (self.X_dot_func.jacobian("x"))
            print "the jacobian wrt to p is %s" % (self.X_dot_func.jacobian("p"))


class RobotModel2D(SystemModel):
    def __init__(self, w_d_n=0, w_r_n=0):
        SystemModel.__init__(self)
        # super(SystemModel, self).__init__(self)
        # numeric vars
        # numeric params
        w_d_n = w_d_n
        w_r_n = w_r_n
        self.p_num = np.array([w_d_n, w_r_n])

        # symbolic vars

        # symbolic params
        w_d = SX.sym("w_d")
        w_r = SX.sym("w_r")
        self.p_sym = vertcat([w_d, w_r])

        # symbolic states
        self.state = SX.sym("x", 3)  # x coordinate, y coordinate, theta (orientation)
        x = self.state  # used for short hand notation during the dev of equations

        # symbolic inputs
        v_l = SX.sym("v_l")
        v_r = SX.sym("v_r")
        self.inputs = vertcat([v_l, v_r])

        # outputs
        self.C = np.diag([1, 1, 1])
        self.outputs = mul(self.C, self.state)

        # casadi function
        x1_dot = cos(x[2])*(v_l+v_r)*0.5
        x2_dot = sin(x[2])*(v_l+v_r)*0.5
        x3_dot = (v_r - v_l)/w_d
        self.X_dot = vertcat([x1_dot, x2_dot, x3_dot])
        self.X_dot_func = SXFunction('f', daeIn(x=self.state, p=vertcat([self.inputs, self.p_sym]), t=self.t_sym),
                                     daeOut(ode=self.X_dot))

    def system_ode_odeint(self, x, t, params):
        x1, x2, x3 = x
        v_l, v_r = params
        x1_dot = cos(x3)*(v_l+v_r)*0.5
        x2_dot = sin(x3)*(v_l+v_r)*0.5
        x3_dot = (v_r - v_l)/self.p_num[0]
        X_dot = [x1_dot, x2_dot, x3_dot]
        return X_dot


class RobotModel3D(RobotModel2D):

    def __init__(self, w_d_n=0, w_r_n=0):
        # super(RobotModel2D, self).__init__(self, w_d_n, w_r_n)
        RobotModel2D.__init__(self, w_d_n, w_r_n)

        # numeric params and symbolic params are the same as RobotModel2D

        # symbolic inputs # gyro readings
        roll_dot = SX.sym("dRoll")
        pitch_dot = SX.sym("dPitch")
        yaw_dot = SX.sym("dYaw")
        additional_inputs = vertcat([roll_dot, pitch_dot, yaw_dot])
        self.inputs = vertcat([self.inputs, additional_inputs])

        # symbolic states
        self.state = SX.sym("x", 6)  # x[0] coordinate, y[1] coordinate, z[2] coordinate, roll[3], pitch[4], yaw[5]
        x = self.state  # used for short hand notation during the dev of equations

        # outputs
        self.C = np.diag([1, 1, 1, 1, 1, 1])
        self.outputs = mul(self.C, self.state)

        # casadi function
        x1_dot = 0.5*(self.inputs[0]+self.inputs[1])*cos(x[5])*cos(x[4])
        x2_dot = 0.5*(self.inputs[0]+self.inputs[1])*sin(x[5])*cos(x[4])
        # note the negative sign for pitch due to the RH coordinate
        # system with x forward y left and z up (ROS convention)
        x3_dot = 0.5*(self.inputs[0]+self.inputs[1])*sin(-x[4])
        x4_dot = roll_dot
        x5_dot = pitch_dot
        x6_dot = (self.inputs[1] - self.inputs[0])*cos(x[3])/self.p_sym[0]
        # x6_dot = yaw_dot  #more accurate
        self.X_dot = vertcat([x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot])
        self.X_dot_func = SXFunction("casadi_ode_function", daeIn(x=self.state,
                                                                  p=vertcat([self.inputs, self.p_sym]),
                                                                  t=self.t_sym), daeOut(ode=self.X_dot))

    def system_ode_odeint(self, x, t, params):
        x1, x2, x3, x4, x5, x6 = x
        v_l, v_r, dRoll, dPitch, dYaw = params

        x1_dot = 0.5*(v_l+v_r)*cos(x6)*cos(x5)
        x2_dot = 0.5*(v_l+v_r)*sin(x6)*cos(x5)
        x3_dot = 0.5*(v_l+v_r)*sin(x5)
        x4_dot = dRoll
        x5_dot = dPitch
        x6_dot = (v_r - v_l)*cos(x4)/self.p_num[0]
        X_dot = [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot]
        return X_dot

        # concatenation snippets
        # print "vertcat([]) gives: %s type %s size %s" % (vertcat([self.p_sym, self.inputs]),
        #                                                  type(vertcat([self.p_sym, self.inputs])),
        #                                                  vertcat([self.p_sym, self.inputs]).shape)  # single vector
        # print "vertcat(()) gives: %s type %s size %s" % (vertcat((self.p_sym, self.inputs)),
        #                                                  type(vertcat((self.p_sym, self.inputs))),
        #                                                  vertcat((self.p_sym, self.inputs)).shape)  # single vectror
        # print "horzcat(()) gives: %s type %s size %s" % (horzcat([self.p_sym, self.inputs]),
        #                                                  type(horzcat([self.p_sym, self.inputs])),
        #                                                  horzcat([self.p_sym, self.inputs]).shape)  # gives a matrix
        # print "np. array() gives: %s type %s size %s" % (np.array((self.p_sym, self.inputs)),
        #                                                  type(np.array((self.p_sym, self.inputs))),
        #                                                  np.array((self.p_sym, self.inputs)).shape)  # array of 2 vec
        # np_1 = np.array([w_d, w_r])
        # np_2 = np.array([v_l, v_r])
        # print "np.append(np.array, np.array) gives: %s type %s size %s" % (np.append(np_1, np_2),
        #                                                                    type(np.append(np_1, np_2)),
        #                                                                    np.append(np_1, np_2).shape)  # vector
