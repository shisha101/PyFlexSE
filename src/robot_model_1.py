import numpy as np
from casadi import *


class SystemModel:
    def __init__(self):
        """
        for the parameter vector make sure that the inputs go in before the parameters(constants/time varying)
        """
        self.p_num = []
        self.p_sym = []
        self.inputs = []
        self.outputs = []
        self.state = []
        # additions
        self.current_state = []
        self.system_equations = None
        self.system_eq_jac = None
        self.X_dot = None
        self.sensor_cov = []
        self.system_cov = []
        self.init_estimate_cov = []
        self.t_sym = SX.sym("t")

    def get_sys_eq(self):
        if self.X_dot is not None:
            return self.X_dot

    def get_p_numeric(self):
        if not self.p_num:
            return self.p_num

    def get_p_symbolic(self):
        if not self.p_sym:
            return self.p_sym

    def get_inputs(self):
        if not self.inputs:
            return self.inputs

    def get_outputs(self):
        if not self.outputs:
            return self.outputs

    def update_system_state(self, new_state):  # addition
        self.current_state = new_state

    def print_debug_casadi(self):  # addition
        if self.system_equations is not None:
            print "the number of input(s) %s output(s) %s " % (self.system_equations.nIn(),self.system_equations.nOut())
            print "the input(s) : %s " %(self.system_equations.symbolicInput())
            print "the output(s): %s " %(self.system_equations.symbolicOutput())
            print "the system input p is %s " %(self.system_equations.getInput("p"))
            print "the jacobian wrt to x is %s" %(self.system_equations.jacobian("x"))
            print "the jacobian wrt to p is %s"%(self.system_equations.jacobian("p"))


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
        self.outputs = self.state

        # casadi function
        x1_dot = cos(x[2])*(v_l+v_r)*0.5
        x2_dot = sin(x[2])*(v_l+v_r)*0.5
        x3_dot = (v_r - v_l)/w_d
        self.X_dot = vertcat([x1_dot, x2_dot, x3_dot])
        self.casadi_function = SXFunction('f', daeIn(x=self.state, p=vertcat([self.inputs, self.p_sym]), t=self.t_sym),
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
        self.outputs = self.state

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
        self.casadi_function = SXFunction("casadi_ode_function", daeIn(x=self.state,
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