import numpy as np
import pandas as pd
import scipy.sparse as sp

from lp_acpr import (create_assignment, create_lp_parameters,
                     format_assignment, lp_solver)


def lp_acpr_test_simple():
    '''
    conditions  papers = 3
                reviewers = 5
                max_reviewers_per_paper = 3
                max_papers_per_reviewer = 2
    '''
    affinity_matrix = np.array([[5, 2, 3, 4, 0],
                                [2, 3, 4, 0, 5],
                                [0, 2, 3, 5, 1]], dtype=np.float)
    a, K, d = create_lp_parameters(affinity_matrix, 3, 2)
    solution = lp_solver(a, K, d)['solution']
    assignment_matrix = create_assignment(solution, affinity_matrix, 0.5)
    format_assignment(assignment_matrix, 'paper', 'reviewers', './outputs/simple_test_result.csv')


def lp_acpr_test_complex():
    '''
    dataset https://gist.github.com/titipata/9c43e3d435d2efb59e394d146e5998e5
    '''
    data = pd.read_csv('./assets/dating_schedule.csv')
    origin_person_id = data['person_id']
    origin_person_id_to_meet = data['person_id_to_meet']
    real_person_id = range(origin_person_id.shape[0])
    pos_dict = {ori: obj for ori, obj in zip(origin_person_id, real_person_id)}
    neg_dict = {obj: ori for ori, obj in zip(origin_person_id, real_person_id)}
    real_person_id_to_meet = []
    for person_id_to_meet in origin_person_id_to_meet:
        real_person_id_to_meet.append([pos_dict[int(person_id)] for person_id in person_id_to_meet.split(';')])
    real_person_id = np.array(real_person_id)
    real_person_id_to_meet = np.array(real_person_id_to_meet)

    affinity_matrix = sp.dok_matrix((data.shape[0], data.shape[0]), dtype=np.float)
    for row in real_person_id:
        affinity_matrix[row, real_person_id_to_meet[row]] = range(len(real_person_id_to_meet[row])-1, -1, -1)

    affinity_matrix = affinity_matrix.toarray()
    a, K, d = create_lp_parameters(affinity_matrix, 5, 5)
    solution = lp_solver(a, K, d)['solution']
    assignment_matrix = create_assignment(solution, affinity_matrix, 0.5)
    format_assignment(assignment_matrix, 'person_id', 'person_id_to_meet', './outputs/complex_test_result.csv')


if __name__ == "__main__":
    lp_acpr_test_simple()
    lp_acpr_test_complex()

    pass
