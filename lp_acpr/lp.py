import numpy as np
from ortools.linear_solver import pywraplp
from scipy.sparse import coo_matrix

__all__ = ['lp_solver']


def lp_solver(objective_function_vector, constraints_matrix, constraints_vector):
    '''
    Reference   https://en.wikipedia.org/wiki/Linear_programming
    Solve the following linear programming problem
            maximize    (c.T).dot(x)
            subject to  A.dot(x) <= b
    Where   x represents the vector of variables
            c are vector of coefficients    # objective_function_vector
            A are matrix of coefficients    # constraints_matrix
            b are vecotr of coefficients    # constraints_vector
    '''
    solver = pywraplp.Solver('lp_solver', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    infinity = solver.infinity()
    cm_row, cm_col = constraints_matrix.shape

    variables = [solver.NumVar(-infinity, infinity, f'x{i}') for i in range(cm_col)]

    constraints = [solver.Constraint(-infinity, int(constraints_vector[i])) for i in range(cm_row)]
    for i, constraint in enumerate(constraints):
        for j in constraints_matrix.col[constraints_matrix.row == i]:
            position = np.logical_and(constraints_matrix.row == i, constraints_matrix.col == j)
            coefficient = constraints_matrix.data[position][0]
            constraint.SetCoefficient(variables[j], coefficient)

    objective_function = solver.Objective()
    for i in range(cm_col):
        objective_function.SetCoefficient(variables[i], objective_function_vector[i])
    objective_function.SetMaximization()

    result_status = solver.Solve()
    solution = np.array([variable.SolutionValue() for variable in variables])

    return {'solution': solution, 'status': result_status}


def lp_test():
    '''
    Reference   http://www.vitutor.com/alg/linear_programming/example_programming.html
    Solve the following linear programming problem
        maximize    f(x,y) = 50x + 40y
        subject to  2x + 3y <= 1500
                    2x + y <= 1000
                    x >= 0 <=> -x <= 0
                    y >= 0 <=> -y <= 0
    '''
    ofv = np.array([50, 40], dtype=np.float)                            # objective_function_vector
    cm = np.array([[2, 3], [2, 1], [-1, 0], [0, -1]], dtype=np.float)   # constraints_matrix
    cv = np.array([1500, 1000, 0, 0])                                   # constraints_vector

    solution = lp_solver(ofv, coo_matrix(cm), cv)
    print(solution)


if __name__ == "__main__":
    lp_test()
