import sys
sys.path.append('/home/abs8rng/catkin_ws/src/my_packages/robot_py_localization/src/casadi-py27-np1.9.1-v2.4.2')
from casadi import *

x = MX.sym("x")
print jacobian(sin(x), x)