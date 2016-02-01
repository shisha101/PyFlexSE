import sys
import numpy as np
sys.path.append('/home/abs8rng/catkin_ws/src/my_packages/robot_py_localization/src/casadi-py27-np1.9.1-v2.4.2')
from casadi import *
import matplotlib.pyplot as plt


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

# timing conditions
t_start = 0.0
t_end = 50.0
delta_t = 0.01
time_array = np.arange(t_start, t_end, delta_t)

# IC
x0_t0 = 1.0
x1_t0 = 0.0
initial_cond = [x0_t0, x1_t0]

#params true

mass_r = 1.0  # mass of cart
sc_r = 1.0    # spring constant
df_r = 0.5   # damping factor
f_r = 5.0     # input force

# parameters symbolic
mass = SX.sym("m")  # mass of cart
sc = SX.sym("sc")   # spring constant
df = SX.sym("df")   # damping factor
F = SX.sym("F")     # input force
x1 = SX.sym("x1")
x2 = SX.sym("x2")
t= SX.sym("t")


x1_dot = x2
x2_dot = -sc*x1/mass - df*x2/mass + F

# X_dot = [x1_dot,x2_dot]
# x = [x1, x2]
# up = [mass, sc, df, F]


X_dot = vertcat((x1_dot,x2_dot))
print type (X_dot)
x = vertcat((x1, x2))
print type(x)
p = vertcat((mass, sc, df, F))
u = F

X_t = np.resize(np.array([]),(len(time_array), x.size()))
X_t[0,:] = initial_cond

f_sim = SXFunction('f', daeIn(x=x,p=p, t=t), daeOut(ode=X_dot)) #
I_options = {}
I_options["t0"] = t_start
I_options["tf"] = t_start+delta_t

integrator = Integrator("ref_integrator", "cvodes", f_sim, I_options)
integrator.setInput(initial_cond,"x0")
integrator.setInput([mass_r,sc_r,df_r,f_r], "p")
# print integrator.getDAE()



for i, time_step in enumerate(time_array):
    integrator.setOption({"t0": time_step, "tf": t_start+delta_t})
    integrator.setInput([mass_r, sc_r, df_r, f_r], "p")
    integrator.setInput(initial_cond,"x0")
    integrator.evaluate()
    initial_cond = integrator.getOutput()[:,0]
    X_t[i, 0] = integrator.getOutput()[0,:]
    X_t[i, 1] = integrator.getOutput()[1,:]


plot(time_array, X_t, 1)
raw_input("press to exit")


#working simulator
sim = Simulator("sim", integrator, time_array)
sim.setInput(initial_cond, "x0")
sim.setInput([mass_r,sc_r,df_r,f_r], "p")
sim.evaluate()
print sim.getOutput()[0,:].T
# print type(sim.getOutput().toArray())
# print type(sim.getOutput(0).T)

raw_input("plot 2")


# integrator.setOption("abstol",1e-10) # tolerance
# integrator.setOption("reltol",1e-10) # tolerance
# integrator.setOption("steps_per_checkpoint",100)
# integrator.setOption("t0",t_start)
# integrator.setOption("tf",t_end)
# integrator.setOption("fsens_abstol",1e-8)
# integrator.setOption("fsens_reltol",1e-8)
# #integrator.setOption("asens_abstol",1e-8)
# #integrator.setOption("asens_reltol",1e-8)
# integrator.setOption("exact_jacobian",True)
# integrator.init()
#
# simulator = Simulator("my_sim",integrator,time_array)
# print simulator
# for t in xrange(len(time_array)):
#     integrator.setOption("tf", t)
# print integrator({'x0':initial_cond, 'p':[mass_r, sc_r, df_r, f_r]})


# params = [mass, sc, df, F]
