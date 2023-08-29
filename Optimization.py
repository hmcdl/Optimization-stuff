from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import Bounds


def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

# ineq_cons = {'type': 'ineq',
#              'fun' : lambda x: np.array([1 - x[0] - 2*x[1],
#                                          1 - x[0]**2 - x[1],
#                                          1 - x[0]**2 + x[1]]),
#              'jac' : lambda x: np.array([[-1.0, -2.0],
#                                          [-2*x[0], -1.0],
#                                          [-2*x[0], 1.0]])}

# eq_cons = {'type': 'eq',
#            'fun' : lambda x: np.array([2*x[0] + x[1] - 1]),
#            'jac' : lambda x: np.array([2.0, 1.0])}

ineq_cons = {'type': 'ineq',
             'fun' : lambda x: np.array([1 - x[0] - 2*x[1],
                                         1 - x[0]**2 - x[1],
                                         1 - x[0]**2 + x[1]])
            #  'jac' : lambda x: np.array([[-1.0, -2.0],
            #                              [-2*x[0], -1.0],
            #                              [-2*x[0], 1.0]])}
             }
eq_cons = {'type': 'eq',
           'fun' : lambda x: np.array([2*x[0] + x[1] - 1]),
           'jac' : lambda x: np.array([2.0, 1.0])}

# x1 = np.linspace(-1, 1, 50)

# X, Y = np.meshgrid(x1, x1)

# ax = plt.subplot(111, projection='3d')

# ax.plot_surface(X, Y, rosen([X, Y])[0])

# plt.show()

# x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
x0 = np.array([0.5, 0])
# bounds = Bounds([0,-1,-1,-2,0], [2,1,1,3,2])
bounds = Bounds([0, -0.5], [1.0, 2.0])
res = minimize(rosen, x0, method='SLSQP', jac='3-point', #
                constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': True}, #
               bounds=bounds)
print(res.x)