import unittest
from MatVecLib import Matrix
from MatVecLib import Vector


class TestMatrix(unittest.TestCase):
    def test_init(self):
        matrix = [[1, 2], [3, 4]]
        m = Matrix(matrix)
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)
        self.assertEqual(m.matrix, matrix)

    def test_transpose(self):
        matrix = [[1, 2], [3, 4]]
        m = Matrix(matrix)
        m_t = m.transpose()
        self.assertEqual(m_t.rows, 2)
        self.assertEqual(m_t.cols, 2)
        self.assertEqual(m_t.matrix, [[1, 3], [2, 4]])

    def test_multiply(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        m3 = m1 * m2
        self.assertEqual(m3.rows, 2)
        self.assertEqual(m3.cols, 2)
        self.assertEqual(m3.matrix, [[19, 22], [43, 50]])

    def test_multiply_incompatible(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6]])
        with self.assertRaises(ValueError):
            m1 * m2

    def test_add(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        m3 = m1 + m2
        self.assertEqual(m3.rows, 2)
        self.assertEqual(m3.cols, 2)
        self.assertEqual(m3.matrix, [[6, 8], [10, 12]])

    def test_subtract(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        m = Matrix.subtract_matrix(m1,m2)
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)
        self.assertEqual(m.matrix, [[-4, -4], [-4, -4]])

    def test_determinant(self):
        m1 = Matrix([[1, 2], [3, 4]])
        det = m1.determinant()
        self.assertEqual(det, -2)

    def test_inverse(self):
        m1 = Matrix([[1, 2], [3, 4]])
        inv = m1.invert_matrix()
        self.assertEqual(inv.matrix, [[-2.0, 1.0], [1.5, -0.5]])

    def test_square(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m_squared = m1 * m1
        self.assertEqual(m_squared.matrix, [[7, 10], [15, 22]])


    def test_get_matrix_from_input(self):
        m = Matrix()
        m._get_matrix_from_input()
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)


    def test_iter(self):
        m = Matrix([[1, 2], [3, 4]])
        it = iter(m)
        self.assertEqual(next(it), [1, 2])
        self.assertEqual(next(it), [3, 4])

    def test_int(self):
        m = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(TypeError):
            int(m)

    def test_str(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(str(m), '[[1, 2], [3, 4]]')

    class TestVector(unittest.TestCase):
        def test_init(self):
            v = Vector([1, 2, 3])
            self.assertEqual(v.vector, [1, 2, 3])

        def test_get_vector_from_input(self):
            v = Vector()
            v.get_vector_from_input()
            self.assertEqual(v.rows, 3)
            self.assertEqual(v.cols, 1)
            # check the input vector

        def test_cross_product(self):
            v1 = Vector([1, 2, 3])
            v2 = Vector([4, 5, 6])
            v3 = v1.cross_product(v2)
            self.assertEqual(v3, [-3, 6, -3])

        def test_dot_product(self):
            v1 = Vector([1, 2, 3])
            v2 = Vector([4, 5, 6])
            self.assertEqual(v1.dot_product(v2), 32)

if __name__ == '__main__':
    unittest.main()
