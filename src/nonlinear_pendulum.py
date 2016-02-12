from casadi.casadi import *
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from robot_model_1 import SystemModel
from Extended_Kalman_Filter import HybridEKF

import cProfile


class PendulumSystem(SystemModel):
    def __init__(self, m=1.0, l=0.5, d=0.1):
        SystemModel.__init__(self)
        # numeric vars
        # numeric constatns
        g = 9.81
        # numeric params
        m_n = m
        l_n = l
        d_n = d
        self.p_num = np.array([m_n, l_n, d_n])

        # symbolic vars

        # symbolic params
        m = SX.sym("m")
        l = SX.sym("l")
        d = SX.sym("d")
        self.p_sym = vertcat([m, l, d])

        # symbolic states
        self.state = SX.sym("x", 2)  # x coordinate, y coordinate, theta (orientation)
        x = self.state  # used for short hand notation during the dev of equations

        # symbolic inputs
        f = SX.sym("f")
        self.inputs = f

        # outputs
        self.outputs = self.state

        # casadi function
        x1_dot = x[1]
        x2_dot = -(g/l)*sin(x[0]) - (x[1]*d)/(m*l**2)
        self.X_dot = vertcat([x1_dot, x2_dot])
        self.X_dot_func = SXFunction('f', daeIn(x=self.state, p=vertcat([self.inputs, self.p_sym]), t=self.t_sym),
                                     daeOut(ode=self.X_dot))


def extract_cov_values(list_of_cov_matrices):
    if list_of_cov_matrices is not None and len(list_of_cov_matrices) > 0:
        list_of_cov = []
        for sub_mat in list_of_cov_matrices:
            diag_elements = np.diag(sub_mat).T
            list_of_cov.append(diag_elements)
    return np.vstack(list_of_cov)


def plot_2d(x, y, fig_nr):
    fig = plt.figure(fig_nr, figsize=(8, 8))

    plot_1 = fig.add_subplot(211)
    plot_1.plot(x, y[:, 0])
    plot_1.set_xlabel("time (s)")
    plot_1.set_ylabel("position Theta (rad)")

    plot_2 = fig.add_subplot(212)
    plot_2.plot(x, y[:, 1])
    plot_2.set_xlabel("time (s)")
    plot_2.set_ylabel("Omega (rad*s^-1)")

    # plot_3 = fig.add_subplot(413)
    # mod_list_theta = map(lambda x: (x%(2*np.pi))*180/np.pi, y[:, 2])
    # plot_3.plot(x, mod_list_theta)
    # plot_3.set_xlabel("time (s)")
    # plot_3.set_ylabel("theta (deg)")
    # # plot_3.set_title("hi")
    #
    # plot_4 = fig.add_subplot(414)
    # plot_4.plot(y[:, 0], y[:, 1])
    # plot_4.set_xlabel("position x (m)")
    # plot_4.set_ylabel("position y (m)")

    fig.tight_layout()
    fig.show()


def plot_sim_vs_estim(x, y1, y2, fig_nr, y3=None):
    fig = plt.figure(fig_nr, figsize=(8, 8))

    if y3 is None:
        plot_number = 211
    else:
        plot_number = 411

    plot_1 = fig.add_subplot(plot_number)
    plot_1.plot(x, y1[:, 0], "b-")
    plot_1.plot(x, y2[:, 0], "r--")
    plot_1.set_xlabel("time (s)")
    plot_1.set_ylabel("position x (m)")
    plot_number += 1

    plot_2 = fig.add_subplot(plot_number)
    plot_2.plot(x, y1[:, 1], "b-")
    plot_2.plot(x, y2[:, 1], "r--")
    plot_2.set_xlabel("time (s)")
    plot_2.set_ylabel("position y (m)")
    plot_number += 1
    
    if y3 is not None:
        plot_3 = fig.add_subplot(plot_number)
        plot_3.plot(x, y3[:, 0], "r")
        plot_3.set_xlabel("time (s)")
        plot_3.set_ylabel("cov theta")
        plot_number += 1
    
        plot_4 = fig.add_subplot(plot_number)
        plot_4.plot(x, y3[:, 1], "r")
        plot_4.set_xlabel("time (s)")
        plot_4.set_ylabel("cov Omega")
        plot_number += 1

    fig.tight_layout()
    fig.show()


def main():
    # model params
    mass = 1.0
    length = 0.5
    damping_factor = 0.05

    # simulation params
    t_start = 0.0
    delta_t = 0.05
    t_end = 5.0
    t_steps_between_measurements = 20  # how frequently do we get a measurement
    start_state = np.array([[-20.0/180.0*np.pi], [0.0]])  # -20 degrees, zero omega

    # EKF params
    model_trust_factor = 0.05  # how much do we trust our model compared to our measurements affects Q and R matrices
    initial_state_trust_factor = 0.001  # how much we trust our initial estimate P0
    initial_state_pre_multiplication_factor = -1.0  # affects the starting state of the estimator X0

    t = np.arange(t_start, t_end, delta_t)  # the end+delta_t is because the sim saves the initial sate

    # system
    pendulum = PendulumSystem(mass, length, damping_factor)

    # system integrator
    opt = {"t0": t_start, "tf": delta_t}
    # non time varying system constant delta t with t0 = 0
    system_integrator = Integrator("real_pendulum_integrator", "cvodes", pendulum.X_dot_func, opt)

    # Hybrid EKF
    hybrid_EKF = HybridEKF(pendulum, 0.05,
                           start_state * initial_state_pre_multiplication_factor,
                           Rin=np.eye(2),
                           Qin=(1 / model_trust_factor) * np.eye(2),
                           P0=(1 / initial_state_trust_factor) * np.eye(2))
    # storage lists
    estimated_states = []
    estimated_cov = []
    real_states = []
    estimated_states.append(start_state * initial_state_pre_multiplication_factor)  # append state at t0
    estimated_cov.append((1 / initial_state_trust_factor) * np.eye(2))  # append cov at t0
    real_states.append(start_state)
    print "the start state of the estimator is %s" % hybrid_EKF.X_k_1_p
    print "the start state of the system is %s" % start_state

    for index, time in enumerate(xrange(t.shape[0]-1)):

        # integrate the system
        system_integrator.setInput(real_states[-1], "x0")
        # system_integrator.setOption({"t0": 0.0, "tf": delta_t})
        system_integrator.setInput(np.append([0.0], pendulum.p_num), "p")
        system_integrator.evaluate()
        system_integ_output = system_integrator.getOutput().toArray()
        real_states.append(system_integ_output)

        # do prediction or correction
        if index != 0 and index % t_steps_between_measurements == 0:
            hybrid_EKF.prediction(np.array([0.0]))
            hybrid_EKF.correction(real_states[-1])
        else:
            hybrid_EKF.prediction(np.array([0.0]))
        # print "***Start***"
        # print "EKF output is %s " % hybrid_EKF.X_k_1_p.toArray().T
        # print "step wise system integrator output is %s" % system_integ_output.T
        # print "***End***"
        estimated_states.append(hybrid_EKF.X_k_1_p.toArray())

        estimated_cov.append(hybrid_EKF.P_k_1_p.toArray())

    reformatted_out = np.hstack(estimated_states)
    real_states_ref = np.hstack(real_states)
    cov_values_vector = extract_cov_values(estimated_cov)
    plot_sim_vs_estim(t, real_states_ref.T, reformatted_out.T, 1, cov_values_vector)

if __name__ == '__main__':
    main()
    # cProfile.run("main()")
    raw_input("press to exit")
    # print system_simulator.getOutput().toArray().T
    # print t
