from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import Bounds
from typing import List, Set, Dict, Tuple



def rosen(x):
    """The Rosenbrock function"""
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen2D(x1, x2):
    return 100*(x2-x1**2)**2+(x1-1)**2

def cantileverBeamArea(x):
    w = x[0]
    t = x[1]
    return 4*t/100*(w/100-t/100)

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

#Constants:
L = 2000
P = 20000
q_a = 10
Tau_a = 90
Sigma_a = 165
E = 21*10**4


def J(x):
    w = float(x[0])
    t = float(x[1])
    return 1/12*(w**4 - (w-2*t)**4)

def Q(x):
    w = float(x[0])
    t = float(x[1])
    return 1/8 * w**3 - 1/8*(w-2*t)**3 

def maxNormal(x):
    w = float(x[0])
    t = float(x[1])
    J_ = J(x)
    sig = P*L*w/(2*J_)
    return (P*L*w/(2*J_) - 165)/165

def maxShear(x):
    w = float(x[0])
    t = float(x[1])
    return (P*Q(x) / (2*J(x)*t) - 90)/90

def maxDeflection(x):
    w = float(x[0])
    t = float(x[1])
    maxDef = P*L**3/(3*E*J(x))
    J_val = J(x)
    return (P*L**3/(3*E*J(x)) - 10)/1000

def dim_relation(x):
    w = float(x[0])
    t = float(x[1])
    return (w - 8*t)/100

def dim_relation2(x):
    w = float(x[0])
    t = float(x[1])
    return (2*t-w)/100

ineq_Max_Normal = {'type': 'ineq', 
                   'fun' :maxNormal
                   }

ineq_max_Shear = {'type': 'ineq', 
                   'fun' :maxShear
                   }

ineq_max_Deflection = {'type': 'ineq', 
                   'fun' :maxDeflection
                   }

ineq_dim_relation = {'type': 'ineq', 
                   'fun' :dim_relation
                   }

ineq_dim_relation2 = {'type': 'ineq', 
                   'fun' :dim_relation2
                   }

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

# x1 = np.linspace(1, 20, 50)
# x2 = np.linspace(0.5, 2, 50)


# X, Y = np.meshgrid(x1, x2)
# Z = cantileverBeamArea(X, Y)

# ax = plt.subplot(111, projection='3d')

# ax.plot_surface(X, Y, Z)

# plt.show()

x0 = np.array([60, 10])
x1 = np.array([200, 90])
A0 = cantileverBeamArea(x1)
J_value = J(x1)
Q_value = Q(x1)
Sigma = maxNormal(x1) + 1
Tau = maxShear(x0) + 1
q = maxDeflection(x0) + 10/1000
dr2 = dim_relation([200,1])
rel = dim_relation(x0)

print (Sigma, Tau, q)

bounds = Bounds([10, 1], [200, 100])
res = minimize(cantileverBeamArea, x0, method='SLSQP', jac='3-point', #
                constraints=[ineq_Max_Normal, ineq_max_Shear, ineq_max_Deflection, ineq_dim_relation, ineq_dim_relation2], #
                options={'ftol': 1e-9, 'disp': True}, #
               bounds=bounds)
print(res.x)
x0 = res.x
A0 = cantileverBeamArea(x0)
J_value = J(x0)
Q_value = Q(x0)
Sigma = maxNormal(x0) + 165
Tau = maxShear(x0) + 90
q = maxDeflection(x0) + 10
rel = dim_relation(x0)
print (Sigma, Tau, q)
a=1