from autograd import value_and_grad
import autograd.numpy as np
from genosolver import minimize
import cvxpy as cp

def fg_qubo(Q):
    def _f(x):
        V = x.reshape(Q.shape[0], -1)
        v = np.sqrt(np.sum(V*V, axis=1))
        V2 = V / v[:,None]
        X = np.dot(V2, V2.T)
        return np.sum(X * Q)
    return value_and_grad(_f)

def random_cut(V):
    u = np.random.randn(V.shape[1])
    z = np.sign(V @ u)
    z[z==0] = 1
    return z

def brute(Q):
    for i in range(2**len(Q)):
        ...

if __name__ == '__main__':
    n = 1000
    Q = np.random.randn(n,n) #np.random.randint(-2, 2+1, size=(n,n))
    Q = Q + Q.T

    fg = fg_qubo(Q)
    x0 = np.random.randn(n,n).reshape(-1)
    res = minimize(fg, x0, np=np)
    print(res)
    x = res.x.reshape(n,n)
    V = x/np.sqrt(np.sum(x*x, axis=1))[:,None]
    X_geno = V.dot(V.T)
    print(f'{V = }')
    print(f'{X_geno = }')
    best = np.ones(n)
    for _ in range(1000):
        curr = random_cut(V)
        if best @ Q @ best > curr @ Q @ curr:
            best = curr
    print(best)

    print()
    X_cvxpy = cp.Variable(Q.shape, symmetric=True)
    constraints = [X_cvxpy >> 0, cp.diag(X_cvxpy) == 1]
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X_cvxpy)),
                  constraints)
    prob.solve()
    print(prob)

    print()
    print(f'{Q = }')
    f_geno = fg(X_geno)[0]
    f_cvxpy = fg(X_cvxpy.value)[0]
    print(f'{f_geno - f_cvxpy = }')
    assert abs(f_geno - f_cvxpy) / (1.+abs(f_geno)) < 1e-4
