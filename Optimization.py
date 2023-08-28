from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rosen(x):
    sum =0
    for i in range(0, len(x)-1):
       sum+= 100.0 * np.power((x[i+1] - np.power(x[i], 2.0)), 2.0) + np.power((1.0 - x[i]), 2.0)
    return sum
def rosen_der(x):

    xm = x[1:-1]

    xm_m1 = x[:-2]

    xm_p1 = x[2:]

    der = np.zeros_like(x)

    for i in range (0, len(x) - 2):
        der[i] = 200*(xm[i] - xm_m1[i]**2) - 400*(xm_p1[i] - xm[i]**2)*xm[i] - 2*(1 - xm[i])
    # der[1:-1] = 200*(xm-np.power(xm_m1, 2.0)) - 400*(xm_p1 - np.power(xm, 2.0))*xm - 2*(1-xm)

    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])

    der[-1] = 200*(x[-1]-x[-2]**2)

    return der

# x1 = np.linspace(-1, 1, 50)

# X, Y = np.meshgrid(x1, x1)

# ax = plt.subplot(111, projection='3d')

# ax.plot_surface(X, Y, rosen([X, Y])[0])

# plt.show()

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen, x0, method='BFGS', jac=rosen_der, #
               options={'disp': True})
print(res.x)