import numpy as np
from casadi import *


class SystemModel:
    def __init__(self):
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
        self.w_d_n = w_d_n
        self.w_r_n = w_r_n
        self.p_num = np.array([self.w_d_n, self.w_r_n])

        # symbolic vars

        # symbolic params
        self.w_d = SX.sym("w_d")
        self.w_r = SX.sym("w_r")
        self.p_sym = np.array([self.w_d, self.w_r])
        self.t_sym = SX.sym("t")

        # symbolic states
        self.x1 = SX.sym("x1")  # x coordinate
        self.x2 = SX.sym("x2")  # y coordinate
        self.x3 = SX.sym("x3")  # theta (orientation)
        self.states = np.array([self.x1, self.x2, self.x3])

        # symbolic inputs
        self.v_l = SX.sym("v_l")
        self.v_r = SX.sym("v_r")
        self.inputs = np.array([self.v_l, self.v_r])

        # outputs
        self.outputs = self.states

        # casadi function
        x1_dot = cos(self.x3)*(self.v_l+self.v_r)*0.5
        x2_dot = sin(self.x3)*(self.v_l+self.v_r)*0.5
        x3_dot = (self.v_r - self.v_l)/self.w_d
        self.X_dot = np.array([x1_dot, x2_dot, x3_dot])
        self.casadi_function = SXFunction('f', daeIn(x=self.states, p=np.append(self.p_sym, self.inputs), t=self.t_sym),
                                          daeOut(ode=self.X_dot))

    def system_ode_odeint(self, x, t, params):
        x1, x2, x3 = x
        v_l, v_r = params
        x1_dot = cos(x3)*(v_l+v_r)*0.5
        x2_dot = sin(x3)*(v_l+v_r)*0.5
        x3_dot = (v_r - v_l)/self.w_d_n
        X_dot = [x1_dot, x2_dot, x3_dot]
        return X_dot


class RobotModel3D(RobotModel2D):

    def __init__(self, w_d_n=0, w_r_n=0):
        # super(RobotModel2D, self).__init__(self, w_d_n, w_r_n)
        RobotModel2D.__init__(self, w_d_n, w_r_n)

        # numeric params and symbolic params are the same as RobotModle2D

        # symbolic inputs # gyro readings
        roll_dot = SX.sym("dRoll")
        pitch_dot = SX.sym("dPitch")
        yaw_dot = SX.sym("dYaw")
        self.inputs = np.append(self.inputs, np.array([roll_dot, pitch_dot, yaw_dot]))

        # symbolic states
        # note that x3 is now the z coordinate
        self.x4 = SX.sym("x4")  # roll
        self.x5 = SX.sym("x5")  # pitch
        self.x6 = SX.sym("x6")  # yaw
        self.states = np.append(self.states, np.array([self.x4, self.x5, self.x6]))

        # outputs
        self.outputs = self.states

        # casadi function
        x1_dot = 0.5*(self.v_l+self.v_r)*cos(self.x6)*cos(self.x5)
        x2_dot = 0.5*(self.v_l+self.v_r)*sin(self.x6)*cos(self.x5)
        # note the negative sign for pitch due to the RH coordinate
        # system with x forward y left and z up (ROS convention)
        x3_dot = 0.5*(self.v_l+self.v_r)*sin(-self.x5)
        x4_dot = roll_dot
        x5_dot = pitch_dot
        x6_dot = (self.v_r - self.v_l)*cos(self.x4)/self.w_d
        self.X_dot = np.array([x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot])
        self.casadi_function = SXFunction("casadi_ode_function", daeIn(x=self.states,
                                                                       p=np.append(self.p_sym, self.inputs),
                                                                       t=self.t_sym), daeOut(ode=self.X_dot))

    def system_ode_odeint(self, x, t, params):
        x1, x2, x3, x4, x5, x6 = x
        v_l, v_r, dRoll, dPitch, dYaw = params

        x1_dot = 0.5*(v_l+v_r)*cos(x6)*cos(x5)
        x2_dot = 0.5*(v_l+v_r)*sin(x6)*cos(x5)
        x3_dot = 0.5*(v_l+v_r)*sin(x5)
        x4_dot = dRoll
        x5_dot = dPitch
        x6_dot = (v_r - v_l)*cos(x4)/self.w_d_n
        X_dot = [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot]
        return X_dot
