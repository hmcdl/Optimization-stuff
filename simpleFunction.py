from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import Bounds
from typing import List, Set, Dict, Tuple

def simpleFunction(x):
    w = float(x[0])
    t = float(x[1])
    return t*(w-t)

def g1(x):
    w = float(x[0])
    t = float(x[1])
    return (w-5)**2 - t



def g2(x):
    w = float(x[0])
    t = float(x[1])
    return (t+w) - 10

def dim_relation(x):
    w = float(x[0])
    t = float(x[1])
    return t-w



g1_ineq = {'type': 'ineq', 
                   'fun' :g1
                   }

g2_ineq = {'type': 'ineq', 
                   'fun' :dim_relation
                   }

# x1 = np.linspace(0, 50, 50)
# x2 = np.linspace(0, 50, 50)
# X, Y = np.meshgrid(x1, x2)
# Z = g1([X, Y])
# Z2 = simpleFunction([X, Y])
# ax = plt.subplot(111, projection='3d')
# # ax.set_zlim(0, 100)
# ax.plot_surface(X, Y, Z)
# ax.plot_surface(X, Y, Z2)

# plt.show()

x0 = np.array([5.5 , 4])
bounds = Bounds([5, 3], [10, 10])

res = minimize(simpleFunction, x0, method='SLSQP', jac='3-point', #
                constraints=[g1_ineq, g2_ineq], #
                options={'ftol': 1e-3, 'disp': True}, #
               bounds=bounds)
print(res.x)
print(simpleFunction([5, 3]))