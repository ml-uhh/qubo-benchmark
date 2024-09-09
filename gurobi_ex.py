import numpy as np
from gurobi_optimods.qubo import solve_qubo

if __name__ == '__main__':
    Q = np.random.randint(-10,10, size=(500,500))
    res = solve_qubo(Q)
    print(res)
    
    Q = np.random.randn(100, 100)
    res = solve_qubo(Q)
    print(res)
