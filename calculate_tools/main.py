from sympy import symbols, cos, sin, Matrix, pi

# Define the symbols for joint angles (in radians), link lengths, and offset
theta1, theta2, theta3, theta4 = symbols('theta1 theta2 theta3 theta4')
a2, a3, a4, d1 = symbols('l_2 l_3 l_4 d_1')

# DH parameters
# alpha (twist angle) for each joint
alpha = [pi/2, 0, 0, 0]
# a (link length) for each joint
a = [0, a2, a3, a4]
# d (link offset) for each joint
d = [d1, 0, 0, 0]
# theta (joint angle) for each joint
theta = [theta1, theta2, theta3, theta4]

# DH Transformation matrix
def DH_matrix(theta, alpha, a, d):
    return Matrix([
        [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
        [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
        [0,           sin(alpha),             cos(alpha),            d],
        [0,           0,                      0,                     1]
    ])

# Calculate the transformation matrix for each joint and the total transformation matrix
T = Matrix.eye(4)
for i in range(4):
    T *= DH_matrix(theta[i], alpha[i], a[i], d[i])

print(T)
