import numpy as np
import numpy.linalg as la
from scipy.linalg import hilbert
from scipy.linalg import invhilbert


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

n_list = [5, 8, 12, 20]
tol = 1e-6

for n in n_list:
    A = hilbert(n)
    Ainv = invhilbert(n)
    b = np.ones(n)
    x0 = np.zeros(n)

    _, iterations, _ = CG(A, b, x0, tol)
    print(f"number of iterations: {iterations}.")
    K = la.norm(A) * la.norm(Ainv)
    print(f"Condition number: {K}.")

