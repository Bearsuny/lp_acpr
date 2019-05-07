import numpy as np

from lp_acpr import create_assignment, create_lp_parameters, lp_solver

if __name__ == "__main__":
    affinity_matrix = np.array([[5, 4, 1, 0],
                                [5, 4, 0, 1]], dtype=np.float)
    a, K, d = create_lp_parameters(affinity_matrix, 3, 2)
    print(a.shape, K.shape, d.shape)
    solution = lp_solver(a, K, d)['solution']
    print(solution)
    assignment_matrix = create_assignment(solution, affinity_matrix, 0.5)
    print(assignment_matrix)
    pass
