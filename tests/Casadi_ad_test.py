# sys.path.append('/home/abs8rng/catkin_ws/src/my_packages/robot_py_localization/src/casadi-py27-np1.9.1-v2.4.2')
from casadi import *
import cProfile

x = MX.sym("x")
print jacobian(sin(x), x)

state_v = SX.sym("X", 6)
output_v = SX.sym("Y", 6)
system_matrix = SX.sym("a", 5, 5)
print ("the system matrix is: ", system_matrix)
print type(state_v[0]), state_v[0].shape, state_v[0]
print type(state_v), state_v.shape, state_v

# SX
x = SX.sym("x", 2, 2)
y = SX.sym("y")
f = 3*x + y  # note that the operations were done element wise (SX)
print type(x), x
print x[0]
print type(f), f
print "----- end of SX -----"

# MX
x = MX.sym("x", 2, 2)
y = MX.sym("y")
f = 3*x + y  # note that the operations where only 2 addition and multiplication
print type(x), x
print x[0, 1]
print type(f), f

# to create a matrix use the list of lists syntax
matrix_example = SX([[0, 1], [2, 4]])
print type(matrix_example), matrix_example

M = SX([[3, 7], [4, 5]])
print M
M = diag(SX([3, 4, 5, 6]))
print M
# to transpose
print M.T
M = SX([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print M, M[0, :], M[:, -2]

# concatenations all vectors are considered vertical
x = SX.sym("x", 5)
y = SX.sym("y", 5)
vcat = vertcat([x, y]) # leads to a longer vector
hcat = horzcat([x, y]) # leads that both vectors create a matrix
print x, y, vcat, hcat
# print vertsplit(vcat, [0, 5, 10])  # starting point,


# differentiation
def sx_diff():
    A = SX.sym("A", 3, 3)
    x = SX.sym("x", 3)
    f = mul(A, x)
    print type(f), f
    print jacobian(f, x)


def mx_diff():  # this need to be verified
    A = MX.sym("A", 3, 3)
    x = MX.sym("x", 3)
    f = mul(A, x) # used for matrix multiplication
    print type(f), f
    print jacobian(f, x)


# creation of functions
x = SX.sym('x')
p = SX.sym('p')
f = x**2
g = log(x)-p
print '***'
print "*******==>", type(g)
nlp = SXFunction("nlp", [x, p], [f, g], {"input_scheme": ['x', 'p'], "output_scheme": ['f', 'g']}) # the IO schemes are not needed but are recommended
res = nlp({'x': 1.1, 'p': 3.3}) # this also works
res = nlp([1.1, 3.3])  # evaluation
print res[0], res[1]
print type(res),type(res[0])


x = MX.sym("x", 2, 2) # a matrix of 4 symbolic vars
y = MX.sym("y") # a scalar variable
# note that the * is element wise
output_1 = x
output_2 = sin(y)*x
print "*******==>", type(output_2)
f = MXFunction('function_mx', [x, y], [output_1, output_2])  # Function(display_name, list of input expressions, list of output expressions) note
print(f)
print f([DMatrix([[1, 1], [2, 2]]), 1])


# finding the jacobian
# X = MX.sym('x', 4)
X = SX.sym('x', 4)
X_dot0 = X[0]**2 + X[1]**2 + X[2]**2 + X[3]**3
X_dot1 = sin(X[0])*cos(X[1]) + sin(X[2])*cos(X[3])
X_dot2 = X[0]*X[1]*X[2] + X[1]*X[2]*X[3] + X[2]*X[3]*X[0]
X_dot3 = X[0]+X[1]+X[2]+X[3]
print type(X_dot0)
# X_dot = [X_dot0, X_dot1, X_dot2, X_dot3] # considerd 4 outputs
X_dot = [vertcat([X_dot0, X_dot1, X_dot2, X_dot3])] # considerd 1 output

print X_dot
# sys_eq = MXFunction("system_ode_equations", [X], X_dot)
X_dot_SX = SXFunction("system_ode_equations", [X], X_dot)
# print type(sys_eq), sys_eq.nIn(), sys_eq.nOut()
print type(X_dot_SX), X_dot_SX.nIn(), X_dot_SX.nOut()
# print sys_eq
print X_dot_SX
X_dot_SX.setInput([2,1,1,0])
X_dot_SX.evaluate()
print X_dot_SX.getOutput()
print "************************************************************"
# final implementation
X = SX.sym('x', 4)
X_dot0 = X[0]**2 + X[1]**2 + X[2]**2 + X[3]**2
X_dot1 = sin(X[0])*cos(X[1]) + sin(X[2])*cos(X[3])
X_dot2 = X[0]*X[1]*X[2] + X[1]*X[2]*X[3] + X[2]*X[3]*X[0]
X_dot3 = X[0]+X[1]+X[2]+X[3]
X_dot = [vertcat([X_dot0, X_dot1, X_dot2, X_dot3])] # considerd 1 output
system_equations = SXFunction("system_ode_equations", [X], X_dot)
print system_equations.nIn(), system_equations.nOut()
system_equations.setInput([2, 1, 1, 0])
system_equations.evaluate()
print system_equations.getOutput()
system_eq_jacobian = system_equations.jacobian()
print system_eq_jacobian, type(system_eq_jacobian)
system_eq_jacobian.setInput([2, 1, 1, 0])
system_eq_jacobian.evaluate()
print system_eq_jacobian.getOutput()
# jacobian_fromSX = SXFunction('jacobian', [X], [system_equations.jac()])
# print jacobian_fromSX

# system_equations.hess(0, 1)
# print system_equations.hess(0, 1)

def system_and_jac_evaluation():
    X = SX.sym('x', 4)
    X_dot0 = X[0]**2 + X[1]**2 + X[2]**2 + X[3]**2
    X_dot1 = sin(X[0])*cos(X[1]) + sin(X[2])*cos(X[3])
    X_dot2 = X[0]*X[1]*X[2] + X[1]*X[2]*X[3] + X[2]*X[3]*X[0]
    X_dot3 = X[0]+X[1]+X[2]+X[3]
    X_dot = [vertcat([X_dot0, X_dot1, X_dot2, X_dot3])] # considerd 1 output
    system_equations = SXFunction("system_ode_equations", [X], X_dot)
    system_eq_jacobian = system_equations.jacobian()
    for i in xrange(100,000):
        # print system_equations.nIn(), system_equations.nOut()
        system_equations.setInput([2, 1, 1, 0])
        system_equations.evaluate()
        # print system_equations.getOutput()

        # print system_eq_jacobian, type(system_eq_jacobian)
        system_eq_jacobian.setInput([2, 1, 1, 0])
        system_eq_jacobian.evaluate()
        # print system_eq_jacobian.getOutput()


# f_preferred = Function('f_preferred', [x, y], [x, sin(y) * x], ["x", "y"], ["output_1", "output_2"])  # Function(display_name, list of input expressions, list of output expressions) note



# cProfile.run("sx_diff()")
# cProfile.run("mx_diff()")
cProfile.run("system_and_jac_evaluation()")
print "Time needed for 100,000 evaluations of function and jacobian"