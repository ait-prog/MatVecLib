import MatVecLib as mt
import pandas as pd

# Create a sample matrix
m1 = mt.Matrix([[1, 2], [3, 4]])

# Create a sample vector
v1 = mt.Vector([1, 2, 3])

# Example usage of Matrix functions
print("Matrix examples:")
print("---------------")
print("Matrix addition:")
m2 = mt.Matrix([[5, 6], [7, 8]])
m3 = m1 + m2
print(pd.DataFrame(m3))

print("Matrix subtraction:")
m3 = m1 - m2
print(pd.DataFrame(m3))

print("Matrix multiplication:")
m3 = m1 * m2
print(pd.DataFrame(m3))

print("Matrix inversion:")
m2 = m1.invert_matrix()
print(pd.DataFrame(m2))

print("Matrix determinant:")
det = m1.determinant()
print(det)

print("Matrix trace:")
trace = m1.trace()
print(trace)

print("Matrix transpose:")
m2 = m1.transpose()
print(pd.DataFrame(m2))

print("Matrix square:")
m2 = m1.square(m1)
print(pd.DataFrame(m2))

# Example usage of Vector functions
print("Vector examples:")
print("--------------")
print("Vector cross product:")
v2 = mt.Vector([4, 5, 6])
v3 = v1.cross_product(v2)
print(v3)

print("Vector dot product:")
dot_product = v1.dot_product(v2)
print(dot_product)

