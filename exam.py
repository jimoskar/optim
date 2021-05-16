"""Some algos that can be useful for checking calculations."""
import numpy as np
import numpy.linalg as la

def CG(A,b,x0,tol):
    """Linear conjugate gradient method."""
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

        print(f"iteration: {iter}")
        print(f"residual: {r_new}\n")
 
    return x, iter, r_new



def armijo(f, grad_f, x0, alpha_0 = 1, c = 1/4, rho = 0.1):
    """One step with Armijo linesearch (gradient descent with backtracking)."""
    alpha = alpha_0
    x = x0
    p = -grad_f(x)
    counter = 1
    while f(x + alpha*p) > f(x) + c*alpha*grad_f(x).T @ p:
        alpha *= rho

        print(f"Iteration of loop: {counter}")
        print(f"alpha: {alpha} \n")
        counter += 1
    x = x + alpha*p
    print(f"x: {x}")
    return x
        


if __name__ == "__main__":

    # Example CG
    A = np.array([[2, -1, -1], [-1, 3, -1], [-1, -1, 2]])
    b = np.array([1, 0, 1])
    x0 = np.zeros(3)   
    tol = 1e-3
    #CG(A, b, x0, tol)

    # Example Armijo
    def f(x):
        return x[1]**4 + 3*x[1]**2 - 4*x[0]*x[1] - 2*x[1] + x[0]**2

    def grad_f(x):
        comp1 = -4*x[1] + 2*x[0]
        comp2 = 4*x[1]**3 + 6*x[1] - 4*x[0] - 2
        return np.array([comp1, comp2])

    x0 = np.array([0,0])
    #x = armijo(f,grad_f,x0)

    # Another example

    def g(x):
        return 2*(x[0] - 3*x[1])**2 + x[1]**4

    def grad_g(x):
        comp1 = 4*(x[0] - 3*x[1])
        comp2 = -12*(x[0] - 3*x[1]) + 4*x[1]**3
        return np.array([comp1, comp2])
    
    x0 = np.array([3, 1])
    x = armijo(g, grad_g, x0)



    
    