
import numpy as np
import numpy.linalg as la

## Problem 1

def f(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x)**2

def grad_f(x):
    first = 200 * (x[1] - x[0**2]) * (-2*x[0]) + 2*(x[0] - 1)
    second = 200 * (x[1] - x[0]**2)
    return np.array([first, second])


## Problem 2

def CG(A,b,x0,tol):
    """Conjugate gradient method."""
    N=len(x0)
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