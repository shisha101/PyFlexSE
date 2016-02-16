import numpy as np
from casadi import *
from robot_model_1 import SystemModel
import matplotlib.pyplot as plt
import cProfile


class predator_preay_discrete(SystemModel):
    def __init__(self):
        SystemModel.__init__(self)

        # parameters (symbolic)
        # none

        # states (symbolic)
        self.state = SX.sym("x", 2)  # predator x1, prey x2
        x = self.state

        # inputs (symbolic)
        self.inputs = SX.sym("u", 1)  # food for prey
        u = self.inputs

        # outputs (symbolic)
        self.outputs = self.inputs

        # system equations
        # x1_k+1 = x1_k - 0.8 * x1_k + 0.4 * x2_k + w1_k noise ignored
        x1_k = x[0] - 0.8 * x[0] + 0.4 * x[1]
        # x2_k+1 = x2_k - 0.4 * x1_k + u_k + w2_k  noise ignored
        x2_k = x[1] - 0.4 * x[0] + u[0]
        self.X_dot = vertcat([x1_k, x2_k])

        # casadi function
        self.X_dot_func = SXFunction("discrete_time_system",
                                     [self.state, self.inputs], [self.X_dot])


def plot_2d(x, y, fig_nr):
    fig = plt.figure(fig_nr, figsize=(8, 8))

    plot_1 = fig.add_subplot(211)
    plot_1.plot(x, y[:, 0])
    plot_1.plot(x, y[:, 1])
    plot_1.set_xlabel("time (step)")
    plot_1.set_ylabel("population mean")

    plot_2 = fig.add_subplot(212)
    plot_2.plot(x, y[:, 1])
    plot_2.set_xlabel("time (step)")
    plot_2.set_ylabel("population variance")

    fig.tight_layout()
    fig.show()


def simulate_sys(system, initial_conditions, input, end_time):

    output = []
    output.append(initial_cond)
    for ts in xrange(end_time):
        output.append(system.X_dot_func([output[-1], input[ts]])[-1].toArray().transpose()[0])
    return output


if __name__ == '__main__':

    initial_cond = np.array([10., 20.])
    end_time = 15
    inputs = np.full((end_time, 1), 1.0)
    # print inputs
    # print type(inputs)
    sys_model = predator_preay_discrete()
    output = simulate_sys(sys_model, initial_cond, inputs, end_time)
    output = np.vstack(output)
    time_steps = np.arange(end_time+1)
    print "the shape is %s the output is %s" % (output[:, 0].shape, output[:, 0])
    print "the shape is %s the output is %s" % (time_steps.shape, time_steps)
    cProfile.run("simulate_sys(sys_model, initial_cond, inputs, end_time)")
    plot_2d(time_steps, output, 1)
    raw_input("press any key to exit")

