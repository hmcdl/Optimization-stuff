from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import Bounds
from typing import List, Set, Dict, Tuple
# import warnings
# import pdb



def cantileverBeamArea(w, t):

    return t*(w-t)

def g1(w, t):
    return (w-5)**2 - t

def g2(w, t):
    return -w +5

def g3(w, t):
    return -t +3

def g4(w, t):

    return (t+w) - 10

def dim_relation(w, t):
    return t-w

Z1 = 0
# functions = [(lambda x, y: x/2 + y - 14, Z1)]

# functions = [
#     (lambda x, y: maxNormal(x,y), Z1,'g1'),(lambda x, y: maxShear(x,y), Z1,'g2'), (lambda x, y: maxDeflection(x,y), Z1,'g3'),
#     (lambda x, y: dim_relation(x,y) , Z1,'g4'), (lambda x, y: dim_relation2(x,y) , Z1,'g5')]
functions = [
    (lambda x, y: g1(x,y), Z1,'g1'), (lambda x, y: g2(x,y), Z1,'g2'), (lambda x, y: g3(x,y), Z1,'g3') , 
    (lambda x, y: g4(x,y), Z1,'g1'), (lambda x, y: dim_relation(x,y), Z1,'g1'),]
X, Y = np.meshgrid(np.linspace(0, 10, 1000), np.linspace(0, 10, 1000))

for f, z,name in functions:
    plt.contour(X, Y, f(X, Y), levels = [0], colors='black')
    plt.contourf(X, Y, f(X, Y), [0, 1000], colors='blue', alpha=0.1)


plt.contour(X, Y, cantileverBeamArea(X, Y), levels=100, linestyles='dashed', colors='black')
plt.title('feasible region')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel('w')
plt.ylabel('t')
plt.show()
