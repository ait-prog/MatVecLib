class Matrix:
    def __init__(self, matrix=None):
        """
        Initialize a matrix object
        :param matrix:A list representing the matrix
        :return:None
                """
        if matrix is None:
            self.rows = int(input("Enter number of rows: "))
            self.cols = int(input("Enter number of columns: "))
            self.matrix = self._get_matrix_from_input
        else:
            self.matrix = matrix
            self.rows = len(matrix)
            self.cols = len(matrix[0])

    def __iter__(self):
        """
        Returns an iterator for the matrix
        :return:return an iterator for the matrix
                        """
        return iter(self.matrix)

    def __add__(self, other):
        """
        Add two matrices together
        :param other: other matrix
        :return:a new matrix with the added result
        """
        if self.rows!= other.rows or self.cols!= other.cols:
           raise ValueError()
        result_matrix = [[self.matrix[i][j] + other.matrix[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result_matrix)

    def __sub__(self, other):
        """
        Subtract two matrices together
        :param other: other matrix
        :return: a new matrix with the subtracted result
        """
        if self.rows!= other.rows or self.cols!= other.cols:
            raise ValueError()
        result_matrix = [[self.matrix[i][j] - other.matrix[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return(result_matrix)


    def __mul__(self, other):
        """
        Multiply two matrices together
        :param other: other matrix
        :return: a new matrix with the multiplied result
        """
        if self.cols != other.rows:
            raise ValueError()
        result_matrix = [[0 for _ in range(other.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result_matrix[i][j] += self.matrix[i][k] * other.matrix[k][j]
        return Matrix(result_matrix)

    def __str__(self):
        """
        convert the matrix to a string
        :return: string representation of the matrix
        """
        return str(self.matrix)

    def _get_matrix_from_input(self):
        """
        get a matrix from input
        :return:matrix from input
        """
        matrix = []
        for i in range(self.rows):
            while True:
                try:
                    row = list(map(int, input().split()))
                    if len(row) != self.cols:
                        raise ValueError(f'error')
                    if any(not (0 <= x <= 5) for x in row):
                        raise ValueError(f'error')
                    matrix.append(row)
                    break
                except ValueError as e:
                    print(e)
        return matrix

    def  divinde(self,B):
        """
        Divide two matrices
        :param B : second matrix
        :return: matrix with divide result
        """
        return(Matrix.multiply_matrix(self, Matrix.invert_matrix(B)))

    def random_matrix(self,min_size=1, max_size=100, min_value=0, max_value=10):
        import random
        rows = random.randint(min_size, max_size)
        cols = random.randint(min_size, max_size)
        matrix = [[random.randint(min_value, max_value) for _ in range(cols)] for _ in range(rows)]
        return Matrix(matrix)

    def invert_matrix(self):
        """
        Invert the matrix
        :return:inverted matrix
        """
        if self.rows != self.cols:
            raise ValueError(f'error')
        n = self.rows
        I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        copy = [row[:] for row in self.matrix]
        for f in range(n):
            if copy[f][f] == 0:
                return "Matrix is not invertible"
            fS = copy[f][f]
            for i in range(n):
                copy[f][i] /= fS
                I[f][i] /= fS
            for i in range(n):
                if f != i:
                    fs2 = copy[i][f]
                    for j in range(n):
                        copy[i][j] -= fs2 * copy[f][j]
                        I[i][j] -= fs2 * I[f][j]
        return Matrix([[I[j][i] for i in range(n)] for j in range(n)])
    def determinant(self):
        """
         find the determinant of a matrix
        :return:determinant of matrix
        """
        if self.rows != self.cols:
            raise ValueError(f'error')
        # det = 0
        if self.rows == 2:
            return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]
        det = 0
        for i in range(self.rows):
            minor = self._minor(0, i)
            det += ((-1) ** i) * self.matrix[0][i] * minor.determinant()
        return det

    def _minor(self, row, col):
         """
          a function to find the minor of a matrix
          param row: rows of the matrix
          param col: columns of the matrix
         :return:minor matrix
         """
         minor_matrix = [row[:] for row in self.matrix]
         minor_matrix.pop(row)
         for i in range(len(minor_matrix)):
             minor_matrix[i].pop(col)
         return Matrix(minor_matrix)

    def trace(self):
        """
        Calculate the trace of the matrix
        :return: the trace of the matrix
        """
        if self.rows != self.cols:
            raise ValueError("Matrix must be square to calculate trace")
        return sum(self.matrix[i][i] for i in range(self.rows))


    def add_matrix(self,B): #+
        """
         Add two matrices together
        :paramB: second matrix
        :return:a new matrix with the added result
        """
        if self.rows != B.rows or self.cols != B.cols:
            raise ValueError(f'error')
        result = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                result[i][j] = self.matrix[i][j] + B.matrix[i][j]
        return result

    def multiply_matrix(self, B): #*
        """
         Multiply two matrices together
        :param mat B: second matrix
        :return: a new matrix with the multiplied result
        """

        if self.cols != B.rows:
            raise ValueError
        result = [[0 for _ in range(B.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(B.cols):
                for k in range(self.cols):
                    result[i][j] += self.matrix[i][k] * B.matrix[k][j]
        return result

    def transpose(self):
        """
        transpose the matrix
        :return:transposed matrix
        """

        if self.rows != self.cols:
            raise ValueError(f'error')
        new_matrix = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                new_matrix[i][j] = self.matrix[j][i]
        return Matrix(new_matrix)

    def  square(self,another):  #degree == 2
        """
         make matrix in second degree
        :param another:another matrix
        :return: squared matrix
        """
        if self.rows != another.rows or self.cols != another.cols:
            raise ValueError(f'error')
        res = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(another.cols):
                for k in range(another.cols):
                    res[i][j] += another.matrix[i][k] * another.matrix[k][j]
        return res

    @staticmethod
    def subtract_matrix(m1, m2):
        """
         subtract two matrices
        :param m1: first matrix
        :param m2: second matrix
        :return: subtracted matrix
        """
        if m1.rows != m2.rows or m1.cols != m2.cols:
            raise ValueError
        result_matrix = [[m1.matrix[i][j] - m2.matrix[i][j] for j in range(m1.cols)] for i in range(m1.rows)]
        return Matrix(result_matrix)



class Vector(Matrix):
    def __init__(self,vector=None):
        """
         initialize a vector
        :param vector: A list representing the vector
        :return: None
        """
        if vector:
            self.vector = vector
        else:
            self.vector = []

    def __getitem__(self, index):
        return self.vector[index]

    def get_vector_from_input(self):
        """
         get a vector from input
        :return: vector from input
        """
        vector = []
        self.cols = 1
        for i in range(self.rows):
            continue


    def cross_product(self, vector2): #vector product
        """
        cross product two vectors
        :param vector2: second vector
        :return: cross product of two vectors
        """
        x = self[1] * vector2[2] - self[2] * vector2[1]
        y = -1*(self[0] * vector2[2] - self[2] * vector2[0])
        z = self[0] * vector2[1] - self[1] * vector2[0]
        res = [x,y,z]
        return res
    def dot_product(self,vector2):   #scalar product
        """
         dot product of two vectors
         :param vector1: first vector
        :param vector2: second vector
        :return: dot product of two vectors
        """
        return self[0]*vector2[0]+self[1]*vector2[1]+self[2]*vector2[2]

