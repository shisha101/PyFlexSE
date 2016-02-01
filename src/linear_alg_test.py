import scipy
import numpy as np

x = np.array([1,2,3,4,5])
I_5 = np.eye(5)
mat_2_2 = np.matrix('1 2; 3 4')
I2 = np.matrix(I_5)


print np.linalg.matrix_rank(mat_2_2)

print type(I2)
print mat_2_2
print type(mat_2_2)
print I_5
print type(I_5)
print x
print type(x)
