import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from robot_py_localization.mass_spring_damper_model import msd_model


def system_function_real(x, t, params):
    x0 = x[0]
    x1 = x[1]
    mass, sc, df, f = params
    x0_dot = x1
    x1_dot = -sc*x0**2 -df*x1 + f
    X_dot = [x0_dot,x1_dot]
    return X_dot


def plot(x, y, fig_nr):
    fig = plt.figure(fig_nr,figsize=(8, 8))

    plot_1 = fig.add_subplot(211)
    plot_1.plot(x, y[:,0])
    plot_1.set_xlabel("time (s)")
    plot_1.set_ylabel("distance (m)")

    plot_2 = fig.add_subplot(212)
    plot_2.plot(x, y[:,1])
    plot_2.set_xlabel("time (s)")
    plot_2.set_ylabel("velocity (m.s^-1)")

    fig.tight_layout()
    fig.show()


def main():
    print "inside main function"
    # timing conditions
    t_start = 0.0
    t_end = 50.0
    delta_t = 0.01
    t = np.arange(t_start, t_end, delta_t)

    # IC
    x0_t0 = 1.0
    x1_t0 = 0.0
    initial_cond = [x0_t0, x1_t0]

    # parameters
    mass = 1.0  # mass of cart
    sc = 1.0    # spring constant
    df = 0.5   # damping factor
    f = 5.0     # input force
    params = [mass, sc, df, f]

    x_t = odeint(system_function_real, initial_cond, t, args=(params,))  # note that the end comma is needed

    plot(t, x_t, 1)
    robot_model = msd_model(initial_cond)
    robot_model.print_params()

    x_t2 = odeint(robot_model.model_ode, initial_cond, t, args=(params,))  # note that the end comma is needed
    plot(t, x_t2, 2)


if __name__ == '__main__':
    print "Starting ode test prog."
    main()
    print "ode solver done"
    raw_input("press to exit")
