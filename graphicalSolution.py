from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import Bounds
from typing import List, Set, Dict, Tuple
# import warnings
# import pdb


# warnings.filterwarnings("error", category=RuntimeWarning)
# определение функции f и числовой константы Z
def f1(x, y):
    return y - x  # пример функции
def p(x, y):
    return x + 4/6*y  # пример функции

L = 2000
P = 20000
q_a = 10
Tau_a = 90
Sigma_a = 165
E = 21*10**4

def cantileverBeamArea(w, t):

    return 4*t*(w-t)

def J(w, t):
    return 1/12*(w**4 - (w-2*t)**4) 

def Q(w, t):
    return 1/8 * w**3 - 1/8*(w-2*t)**3 - 90

def maxNormal(w, t):
    J_ = J(w, t)
    return P*L*w/(2*J_) - 165

def maxShear(w, t):
    return P*Q(w, t) / (2*J(w, t)*t) - 90

def maxDeflection(w, t):
    return P*L**3/(3*E*J(w, t)) - 10

def dim_relation(w, t):
    return w - 8*t

def dim_relation2(w, t):
    return 2*t-w

Z1 = 0
# functions = [(lambda x, y: x/2 + y - 14, Z1)]

# functions = [
#     (lambda x, y: maxNormal(x,y), Z1,'g1'),(lambda x, y: maxShear(x,y), Z1,'g2'), (lambda x, y: maxDeflection(x,y), Z1,'g3'),
#     (lambda x, y: dim_relation(x,y) , Z1,'g4'), (lambda x, y: dim_relation2(x,y) , Z1,'g5')]
functions = [
    (lambda x, y: maxNormal(x,y), Z1,'g1'), (lambda x, y: maxShear(x,y), Z1,'g2'), (lambda x, y: maxDeflection(x,y), Z1,'g3'),
    (lambda x, y: dim_relation(x,y) , Z1,'g4'), (lambda x, y: dim_relation2(x,y) , Z1,'g5')]
print(maxShear(100,20))
X, Y = np.meshgrid(np.linspace(1, 300, 1000), np.linspace(1, 50, 1000))

for f, z,name in functions:
    plt.contour(X, Y, f(X, Y), levels = [0], colors='black')
    plt.contourf(X, Y, f(X, Y), [0, 1000000], colors='blue', alpha=0.1)


plt.contour(X, Y, cantileverBeamArea(X, Y), levels=100, linestyles='dashed', colors='black')
plt.title('feasible region')
plt.xlim(0, 300)
plt.ylim(1, 50)
plt.xlabel('w')
plt.ylabel('t')
plt.show()
