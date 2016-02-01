import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from robot_model_1 import RobotModel2D, RobotModel3D
from casadi import *
# import cProfile

print "inside main function"


def plot_2d(x, y, fig_nr):
    fig = plt.figure(fig_nr, figsize=(8, 8))

    plot_1 = fig.add_subplot(411)
    plot_1.plot(x, y[:, 0])
    plot_1.set_xlabel("time (s)")
    plot_1.set_ylabel("position x (m)")

    plot_2 = fig.add_subplot(412)
    plot_2.plot(x, y[:, 1])
    plot_2.set_xlabel("time (s)")
    plot_2.set_ylabel("position y (m)")

    plot_3 = fig.add_subplot(413)
    mod_list_theta = map(lambda x: (x%(2*np.pi))*180/np.pi, y[:, 2])
    plot_3.plot(x, mod_list_theta)
    plot_3.set_xlabel("time (s)")
    plot_3.set_ylabel("theta (deg)")
    # plot_3.set_title("hi")

    plot_4 = fig.add_subplot(414)
    plot_4.plot(y[:, 0], y[:, 1])
    plot_4.set_xlabel("position x (m)")
    plot_4.set_ylabel("position y (m)")

    fig.tight_layout()
    fig.show()


def plot_3d(x, y, fig_nr):
    fig = plt.figure(fig_nr, figsize=(8, 18))

    plot_1 = fig.add_subplot(711)
    plot_1.plot(x, y[:, 0])
    plot_1.set_xlabel("time (s)")
    plot_1.set_ylabel("position x (m)")

    plot_2 = fig.add_subplot(712)
    plot_2.plot(x, y[:, 1])
    plot_2.set_xlabel("time (s)")
    plot_2.set_ylabel("position y (m)")

    plot_3 = fig.add_subplot(713)
    plot_3.plot(x, y[:, 2])
    plot_3.set_xlabel("time (s)")
    plot_3.set_ylabel("position z (m)")
    # plot_3.set_title("hi")

    plot_4 = fig.add_subplot(714)
    mod_list_theta = map(lambda x: (x%(2*np.pi))*180/np.pi, y[:, 3])
    plot_4.plot(x, mod_list_theta)
    plot_4.set_xlabel("time (s)")
    plot_4.set_ylabel("roll (deg)")

    plot_5 = fig.add_subplot(715)
    mod_list_theta = map(lambda x: (x%(2*np.pi))*180/np.pi, y[:, 4])
    plot_5.plot(x, mod_list_theta)
    plot_5.set_xlabel("time (s)")
    plot_5.set_ylabel("pitch (deg)")

    plot_6 = fig.add_subplot(716)
    mod_list_theta = map(lambda x: (x%(2*np.pi))*180/np.pi, y[:, 5])
    plot_6.plot(x, mod_list_theta)
    plot_6.set_xlabel("time (s)")
    plot_6.set_ylabel("yaw (deg)")

    plot_7 = fig.add_subplot(717)
    plot_7.plot(y[:, 0], y[:, 1])
    plot_7.set_xlabel("position x (m)")
    plot_7.set_ylabel("position y (m)")
    plot_7.set_title("top view")

    fig.tight_layout()
    fig.show()


# def odeint_function():
#     x_t = odeint(robot_model_2D.system_ode_odeint, initial_cond, t, args=(params,))  # note that the end comma is needed


# def casadi_function():
#
#     I_options = {}
#     I_options["t0"] = t_start
#     I_options["tf"] = t_end
#
#     casadi_integtrator = Integrator("2dRobot", "cvodes", robot_model_2D.casadi_function, I_options)
#     # casadi_integtrator.setInput(initial_cond, "x0")
#     # casadi_integtrator.setInput(np.append(robot_model.p_num, params), "p")
#     # casadi_integtrator.evaluate()
#     sim = Simulator("sim", casadi_integtrator, t)
#     sim.setInput(initial_cond, "x0")
#     sim.setInput(np.append(robot_model_2D.p_num, params), "p")
#     sim.evaluate()
#     # print (t.shape, "shape of t")
#     # print (sim.getOutput().toArray().T.shape, "shape of array")
#     # plot(t, sim.getOutput().toArray().T, 2)


def main():
    # timing conditions
    t_start = 0.0
    t_end = 50.0
    delta_t = 0.001
    t = np.arange(t_start, t_end, delta_t)

    # IC
    x1_t0 = 0.0
    x2_t0 = 0.0
    x3_t0 = 0.0
    initial_cond = [x1_t0, x2_t0, x3_t0]

    # # parameters
    v_l = 0.5
    v_r = 1.0
    params = [v_l, v_r]

    robot_model_2D = RobotModel2D(1.5, 0.33)

    x_t = odeint(robot_model_2D.system_ode_odeint, initial_cond, t, args=(params,))  # note that the end comma is needed
    plot_2d(t, x_t, 1)

    I_options = {}
    I_options["t0"] = t_start
    I_options["tf"] = t_end

    casadi_integrator = Integrator("2dRobot", "cvodes", robot_model_2D.casadi_function, I_options)
    casadi_integrator.setInput(initial_cond, "x0")
    casadi_integrator.setInput(np.append(robot_model_2D.p_num, params), "p")
    casadi_integrator.evaluate()
    sim = Simulator("sim", casadi_integrator, t)
    sim.setInput(initial_cond, "x0")
    sim.setInput(np.append(robot_model_2D.p_num, params), "p")
    sim.evaluate()
    print (t.shape, "shape of t")
    print (sim.getOutput().toArray().T.shape, "shape of array")
    plot_2d(t, sim.getOutput().toArray().T, 2)

    # IC 3D
    x1_t0 = 0.0
    x2_t0 = 0.0
    x3_t0 = 0.0
    x4_t0 = 0.0
    x5_t0 = 0.0
    x6_t0 = 0.0
    initial_cond = [x1_t0, x2_t0, x3_t0, x4_t0, x5_t0, x6_t0]

    # parameters 3D odeint
    v_l = 0.9
    v_r = 1.0
    dRoll = 0.02
    dPitch = 0.0
    dYaw = 0.0
    params = [v_l, v_r, dRoll, dPitch, dYaw]

    robot_model_3D = RobotModel3D(1.5, 0.33)
    x_t = odeint(robot_model_3D.system_ode_odeint, initial_cond, t, args=(params,))  # note that the end comma is needed
    plot_3d(t, x_t, 3)

    # Casadi 3D

    # parameters 3D odeint
    v_l = 0.9
    v_r = 1.0
    dRoll = 0.02
    dPitch = 0.0
    dYaw = 0.0
    params = [v_l, v_r, dRoll, dPitch, dYaw]

    I_options = {}
    I_options["t0"] = t_start
    I_options["tf"] = t_end

    casadi_integrator = Integrator("2dRobot", "cvodes", robot_model_3D.casadi_function, I_options)
    casadi_integrator.setInput(initial_cond, "x0")
    casadi_integrator.setInput(np.append(robot_model_3D.p_num, params), "p")
    casadi_integrator.evaluate()
    sim = Simulator("sim", casadi_integrator, t)
    sim.setInput(initial_cond, "x0")
    sim.setInput(np.append(robot_model_3D.p_num, params), "p")
    sim.evaluate()
    plot_3d(t, sim.getOutput().toArray().T, 4)


if __name__ == '__main__':
    print "Starting ode test prog."
    main()
    # cProfile.run("odeint_function()")
    # cProfile.run("casadi_function()")
    print "ode solver done"
    print "end of test !"
    raw_input("press to exit")
