import unittest
from casadi import *
import numpy as np


class TestCasADi(unittest.TestCase):
    def setUp(self):
        self.x = SX.sym("x")
        self.y = SX.sym("y")
        self.sym_vector_4_1 = SX.sym("v", 4, 1)
        self.sym_matrix_4_4 = SX.sym("m", 4, 4)
        self.sym_matrix_4_4_2 = SX.sym("m", 4, 4)
        self.sym_vector_1_4 = SX.sym("v", 1, 4)
        self.matrix_one_4_4 = DMatrix(np.array([[1., 1., 1., 1.],
                                                [1., 1., 1., 1.],
                                                [1., 1., 1., 1.],
                                                [1., 1., 1., 1.]]))
        self.I = SX.eye(4)
        self.scalar = 1.0
        self.ones_array = np.ones((4, 1))
        self.ones_matrix = np.ones((4, 4))

    def test_that_tests_work(self):
        self.failUnless(True)

    def test_add_two_syms(self):
        result = self.x + self.y
        result_subs = substitute(result, vertcat([self.x, self.y]), vertcat([1.0, 1.0]))
        # result_subs = substitute(result, vertcat([self.x, self.y]), np.array([1.0, 1.0]))
        # result_subs = substitute(result, vertcat([self.x, self.y]), [1.0, 1.0])
        self.assertEqual(result_subs, 2.0)

    def test_SXobj_substitute(self):
        result = substitute(self.sym_vector_4_1, self.sym_vector_4_1, vertcat([1.0, 1.0, 1.0, 1.0]))
        result_funciton = SXFunction("f", [self.sym_vector_4_1], [result])  # np array needed to get rid of Dmatrix
        result_funciton_out = result_funciton([vertcat([1.0, 1.0, 1.0, 1.0])])[-1].toArray()
        check = np.array([1., 1., 1., 1.]).transpose() == result_funciton_out
        self.failUnless(check.all() == True)

    def test_SXFunction_substitute(self):
        result_funciton = SXFunction("f", [self.sym_vector_4_1], [vertcat([1.0, 1.0, 1.0, 1.0])])
        # np array needed to get rid of Dmatrix
        result_funciton_out = result_funciton([vertcat([1.0, 1.0, 1.0, 1.0])])[-1].toArray()
        check = self.ones_array.transpose() == result_funciton_out
        self.failUnless(check.all() == True)

    def test_add_vector_to_scalar(self):
        result = self.sym_vector_1_4 + self.scalar
        # print type(result)
        result_subs = substitute(result, self.sym_vector_1_4, horzcat([1., 1., 1., 1.]))
        result_func = SXFunction("f", [self.sym_vector_1_4], [result_subs])
        result_num = result_func([horzcat([1., 1., 1., 1.])])[-1].toArray()
        check = self.ones_array.transpose()+self.scalar == result_num
        self.failUnless(check.all() == True)

    def test_add_matrix_to_scalar(self):
        result = self.sym_matrix_4_4 + self.scalar
        result_func = SXFunction("f", [self.sym_matrix_4_4], [result])
        result_num = result_func([self.matrix_one_4_4])[-1].toArray()
        # print result_num
        check = result_num == self.ones_matrix+self.scalar
        self.failUnless(check.all() == True)

    def test_add_vector_to_scalar_var(self):
        result = self.sym_vector_1_4 + self.x
        # print type(result)
        result_func = SXFunction("f", [self.sym_vector_1_4, self.x], [result])
        result_num = result_func([horzcat([1., 1., 1., 1.]), self.scalar])[-1].toArray()
        # print result_num
        # print self.ones_array.transpose()+self.scalar
        check = self.ones_array.transpose()+self.scalar == result_num
        # print check
        self.failUnless(check.all() == True)

    def test_add_matrix_to_scalar_var(self):
        result = self.sym_matrix_4_4 + self.x + self.y
        result_func = SXFunction("f", [self.sym_matrix_4_4, self.x, self.y], [result])
        result_num = result_func([self.matrix_one_4_4, self.scalar, self.scalar])[-1].toArray()
        # print result_num
        check = self.ones_matrix+self.scalar+self.scalar == result_num
        self.failUnless(check.all() == True)

    def test_add_vectors(self):
        result = self.sym_vector_4_1.T + self.sym_vector_1_4
        result_func = SXFunction("f", [self.sym_vector_4_1, self.sym_vector_1_4], [result])
        result_num = result_func([self.ones_array.transpose(), self.ones_array])[-1].toArray()
        check = result_num == 2*self.ones_array
        self.failUnless(check.all() == True)

    def test_add_vectors_bad_dim(self):
        try:
            result = self.sym_vector_1_4 + self.sym_vector_4_1
        except Exception as e:
            self.assertEqual(e.__class__.__name__, "RuntimeError")

    def test_add_matrix(self):
        result = self.sym_matrix_4_4 + self.sym_matrix_4_4_2
        result_func = SXFunction("f", [self.sym_matrix_4_4, self.sym_matrix_4_4_2], [result])
        result_num = result_func([self.ones_matrix, self.ones_matrix])[-1].toArray()
        check = result_num == 2 * self.ones_matrix
        self.assertTrue(check.all())

    def test_add_matrix_to_vector(self):
        try:
            result = self.sym_matrix_4_4 + self.sym_vector_4_1
        except Exception as ex:
            # print ex.__class__.__name__
            self.assertTrue(ex.__class__.__name__ == "RuntimeError")

    def test_matrix_size(self):
        self.assertEqual(self.sym_matrix_4_4.shape, (4, 4))

    def test_vector_size(self):
        self.assertTrue(self.sym_vector_4_1.shape, (4, 1))

    def test_vector_size_2(self):
        self.assertTrue(self.sym_vector_1_4.shape, (1, 4))

    def test_vector_multiplication_4_1_mul_1_4_size(self):
        result = mul(self.sym_vector_4_1, self.sym_vector_1_4)
        self.assertEqual(result.shape, (4, 4))

    def test_vector_multiplication_4_1_mul_1_4_value(self):
        result = mul(self.sym_vector_4_1, self.sym_vector_1_4)
        result_func = SXFunction("f", [self.sym_vector_4_1, self.sym_vector_1_4], [result])
        result_num = result_func([self.ones_array*self.scalar, self.ones_array.transpose()])[-1].toArray()
        check = result_num == self.ones_matrix*self.scalar
        self.assertTrue(check.all())

    def test_vector_multiplication_4_1_mul_4_1(self):
        try:
            result = mul(self.sym_vector_4_1, self.sym_vector_4_1)
        except Exception as e:
            self.assertEqual(e.__class__.__name__, "RuntimeError")

    def test_matrix_multiplication_4_4_mul_4_4_value(self):
        result = mul(self.sym_matrix_4_4, self.sym_matrix_4_4_2)
        result_func = SXFunction("f", [self.sym_matrix_4_4, self.sym_matrix_4_4_2], [result])
        result_num = result_func([self.ones_matrix, self.ones_matrix])[-1].toArray()
        check = result_num == self.ones_matrix*self.ones_matrix.shape[0]
        # print self.sym_matrix_4_4 * self.sym_matrix_4_4_2  # element wise multiplication
        self.assertTrue(check.all())

    def test_matrix_multiplication_4_4_mul_4_4_shape(self):
        result = mul(self.sym_matrix_4_4, self.sym_matrix_4_4_2)
        self.assertEqual(result.shape, (4, 4))

    def test_SXFunction_evaluate(self):
        result = mul(self.sym_matrix_4_4, self.sym_matrix_4_4_2)
        result_func = SXFunction("f", [self.sym_matrix_4_4, self.sym_matrix_4_4_2], [result])
        result_num = result_func([self.ones_matrix, self.ones_matrix])[-1].toArray()
        check = result_num == self.ones_matrix*self.ones_matrix.shape[0]
        self.assertTrue(check.all())

    def test_SXobj_evaluate(self):
        result = mul(self.sym_matrix_4_4, self.sym_matrix_4_4_2)  # multiplication
        result_sub = substitute(result, vertcat([self.sym_matrix_4_4]), vertcat([self.matrix_one_4_4]))  # subs
        # creating function from exp that was substituted in
        result_func = SXFunction("f", [self.sym_matrix_4_4_2], [result_sub])
        result_num = result_func([self.matrix_one_4_4])[-1].toArray()
        check = result_num == self.ones_matrix*self.ones_matrix.shape[0]
        self.assertTrue(check.all())
        self.failUnless(True)


class TestJacobiansCasADi(unittest.TestCase):
    def setUp(self):
        self.A = SX.sym("a", 3, 3)
        self.X = SX.sym("x", 3, 1)
        self.ones_array = np.ones((3, 1))
        self.ones_matrix = np.ones((3, 3))
        self.SXobj_mat_mul_vec = mul(self.A, self.X)
        # manual jacobian and evaluation of A * x for A and x all having elements of 1
        self.full_jacobian_check_matrix = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
                                                   [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1],
                                                   [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]])

        self.SXfunc_mat_mul_vec = SXFunction("mat_mul_vector", [self.A, self.X], [self.SXobj_mat_mul_vec],
                                             {"input_scheme": ["A", "X"], "output_scheme": ['out']})

    # def print_results(self):
    #     print " \n the type of the SXobj is %s and the contents are %s" % \
    #           (type(self.SXobj_mat_mul_vec), self.SXobj_mat_mul_vec)
    #     print "the number of inputs are %s, the number of outputs are %s" % \
    #           (self.SXfunc_mat_mul_vec.nIn(), self.SXfunc_mat_mul_vec.nOut())
    #     print "the function inputs are %s, the function outputs are %s and the output expression is \n %s" % \
    #           (self.SXfunc_mat_mul_vec.inputScheme(), self.SXfunc_mat_mul_vec.outputScheme(),
    #            self.SXfunc_mat_mul_vec.outputExpr())
    #     print "using fullJacobian we get the type %s" % (type(self.SXfunc_mat_mul_vec.fullJacobian()))
    #     print "using jacobian we get the type %s" % (type(self.SXfunc_mat_mul_vec.jacobian()))
    #     print "using jac we get the type %s" % (type(self.SXfunc_mat_mul_vec.jac()))
    #     self.SXfunc_mat_mul_vec.printDimensions()
    #     print "*******************"
    #     print self.SXfunc_mat_mul_vec
    #     self.SXfunc_mat_mul_vec.printDimensions()  # get the functions Dimensions and variable names
    #     print self.SXfunc_mat_mul_vec.outputExpr()  # get the output expression in readable format
    #     print "*******************"
    #     self.assertTrue(True)

    def test_SXFunction_jacobian_complete(self):
        full_func_jacob = self.SXfunc_mat_mul_vec.fullJacobian()
        # full_func_jacob.printDimensions()
        # returns dict because dict was used for calling
        result = full_func_jacob({"A": self.ones_matrix, "X": self.ones_array})["jac"].toArray()
        check = result == self.full_jacobian_check_matrix
        # print check
        self.assertTrue(check.all())

    def test_SXobj_jacobian_complete(self):
        full_jacobian = jacobian(self.SXobj_mat_mul_vec, horzcat([self.A, self.X]))
        func_for_eval = SXFunction("f", [self.A, self.X], [full_jacobian])
        result = func_for_eval([self.ones_matrix, self.ones_array])[-1].toArray()
        check = result == self.full_jacobian_check_matrix
        self.assertTrue(check.all())

    def test_SXFunction_jacobian_wrt_one_var_1(self):
        jac_wrt_A = self.SXfunc_mat_mul_vec.jacobian("A")
        result = jac_wrt_A([self.ones_matrix, self.ones_array])[0].toArray()
        check = result == self.full_jacobian_check_matrix[0:3, 0:9]
        # jac_wrt_A.printDimensions()
        self.assertTrue(check.all())

    def test_SXFunction_jacobian_wrt_one_var_2(self):
        jac_wrt_X = self.SXfunc_mat_mul_vec.jacobian("X")
        result = jac_wrt_X([self.ones_matrix, self.ones_array])[0].toArray()
        check = result == self.full_jacobian_check_matrix[0:3, 9::]
        # jac_wrt_A.printDimensions()
        self.assertTrue(check.all())

if __name__ == '__main__':
    # unittest.main()
    suite_casadi = unittest.TestLoader().loadTestsFromTestCase(TestCasADi)
    unittest.TextTestRunner(verbosity=2).run(suite_casadi)
    suite_casadi_jac = unittest.TestLoader().loadTestsFromTestCase(TestJacobiansCasADi)
    unittest.TextTestRunner(verbosity=2).run(suite_casadi_jac)
