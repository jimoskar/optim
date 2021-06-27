%matplotlib inline
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy.linalg as la

## Problem 1

def f(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x)**2

def grad_f(x):
    first = 200 * (x[1] - x[0**2]) * (-2*x[0]) + 2*(x[0] - 1)
    second = 200 * (x[1] - x[0]**2)
    return np.array([first, second])


def wolfeLineSearch(f, g, xk, fk, gk, pk, ak, c1, c2, rho, nmaxls=100):
    
    pkgk = np.dot(pk, gk)
    
    # Test if the conditions are satisfied for ak = 1
    if f(xk+pk) <= fk + c1*pkgk and np.dot(pk, g(xk+pk)) > c2*pkgk:
        return 1
    
    # Increase the step length until the Armijo rule is (almost) not satisfied
    while f(xk + rho*ak*pk) <= fk + c1*rho*ak*pkgk:
        ak *= rho
    
    # Use bisection to find the optimal step length
    aU = ak # upper step length limit
    aL = 0  # lower step length limit
    for i in range(nmaxls):
        
        # Find the midpoint of aU and aL
        ak = 0.5*(aU + aL)
        
        if f(xk+ak*pk) > fk + c1*ak*pkgk:
            # Armijo condition is not satisfied, decrease the upper limit
            aU = ak
            continue
        
        if np.dot(pk, g(xk+ak*pk)) > -c2*pkgk:
            # Upper Wolfe condition is not satisfied, decrease the upper limit
            aU = ak
            continue
        
        if np.dot(pk, g(xk+ak*pk)) < c2*pkgk:
            # Lower Wolfe condition is not satisfied, increase the lower limit
            aL = ak
            continue
            
        # Otherwise, all conditions are satisfied, stop the search
        break
    
    return ak

def searchDirection(xk, gk, H, method, epsilon=1e-8):
    
    if method == 'newton':
        pk = -np.linalg.solve(H(xk),gk) # compute the search direction
        if -np.dot(pk,gk) <= epsilon*np.linalg.norm(gk)*np.linalg.norm(pk): 
            return -gk # ensure that the directional derivative is negative in this direction
        return pk
    
    if method == 'gd':
        return -gk
    
    raise('Method ' + str(method) + 'unknown.')

def optimize(f, g, H, x0, method='newton', tol=1e-8, nmax=1000, nmaxls=100, c1=1e-4, c2=0.9, rho=2, epsilon=1e-8):
    
    xk = x0[None,:]  # list of all the points
    fk = f(xk[-1,:]) # current function value
    gk = g(xk[-1,:]) # current gradient
    ak = 1           # current step length
    
    for k in range(nmax):
        
        # Compute the search direction
        pk = searchDirection(xk[-1,:], gk, H, method, epsilon)
        
        # Perform line search
        ak = wolfeLineSearch(f, g, xk[-1,:], fk, gk, pk, ak, c1, c2, rho, nmaxls)
        
        # Perform the step, add the step to the list, and compute the f and the gradient of f for the next step
        xk = np.append(xk, xk[[-1],:] + ak*pk, axis=0)
        fk = f(xk[-1,:])
        gk = g(xk[-1,:])
        
        if np.linalg.norm(gk) < tol:
            break
    
    return xk

# Objective function and its gradient and Hessian
f = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2 
g = lambda x: np.array([
        -400*(x[1]-x[0]**2)*x[0] - 2*(1-x[0]), 
        200*(x[1]-x[0]**2)
    ])
H = lambda x: np.array([
        [-400*x[1]+1200*x[0]**2+2, -400*x[0]],
        [-400*x[0], 200]
    ])
xt = [[1,1]]


# Initial conditions
x0 = np.array([0.5,0.5])

# Sequences produced by the methods
xnewton = optimize(f, g, H, x0, method='newton')
xgd = optimize(f, g, H, x0, method='gd')

knewton = np.arange(xnewton.shape[0])
kgd = np.arange(xgd.shape[0])

# Plot the results
plt.figure(figsize=(10,8))

# Plot the evolution of the methods
X,Y = np.meshgrid(np.linspace(-3,3,200), np.linspace(-1,2,200))
Z = np.apply_along_axis(f,2,np.concatenate((X[:,:,None],Y[:,:,None]),axis=2))
ax1 = plt.subplot(211)
ax1.contourf(X,Y,Z,levels=np.linspace(0,10,20)**5, cmap='Blues', locator=ticker.LogLocator())
ax1.plot(xnewton[:,0], xnewton[:,1], label='Newton\'s method')
ax1.plot(    xgd[:,0],     xgd[:,1], label='Gradient descent')
ax1.set_aspect('equal')
ax1.legend()

# Backward error
ax2 = plt.subplot(223)
ax2.plot(knewton,np.linalg.norm(xnewton-xt,axis=1), label='Newton\'s method')
ax2.plot(kgd,np.linalg.norm(    xgd-xt,axis=1), label='Gradient descent')
ax2.legend()
plt.xlabel('$k$')
plt.title('$|x_k-x^*|$')
plt.yscale('log')

# Forward error
ax3 = plt.subplot(224)
ax3.plot(knewton,np.apply_along_axis(f, 1, xnewton), label='Newton\'s method')
ax3.plot(kgd,np.apply_along_axis(f, 1,     xgd), label='Gradient descent')
ax3.legend()
plt.xlabel('$k$')
plt.title('$|f(x_k)-f(x^*)|$')
plt.yscale('log')

plt.show()

## Problem 2

def CG(A,b,x0,tol):
    """Conjugate gradient method."""
    N = len(x0)
    x = x0
    r_new = A @ x - b
    p = - r_new
    iter = 0
    while (la.norm(r_new) > tol and iter < N):

        r_prev = r_new
        rr = r_prev.T @ r_prev
        alpha = rr / (p.T @ A @ p)
        x = x + alpha * p
        r_new = r_prev + alpha * A @ p
        beta = (r_new.T @ r_new) / rr
        p = -r_new + beta * p
        
        iter += 1
 
    return x, iter, r_new

A = np.array([[2, -1, -1], [-1, 3, -1], [-1, -1, 2]])
b = np.array([1, 0, 1])
x0 = np.zeros(3)
print(CG(A, b, x0, 1e-5))