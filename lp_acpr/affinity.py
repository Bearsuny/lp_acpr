import numpy as np
import scipy.sparse as sp

__all__ = ['create_lp_parameters', 'create_assignment']


def create_lp_parameters(affinity_matrix, max_reviewers_per_paper=10, max_papers_per_reviewer=10):
    '''
    Reference   Taylor, Camillo J. "On the optimal assignment of conference papers to reviewers." (2008).
    The problem formulation of assignment of conference papers to reviewers is as follow
            maximize    (a.T).dot(b)
            subject to  K.dot(b) <= d
                        K = [N_p; N_r; I; -I] and d = [c_p; c_r, 1; 0]
    where   a is the affinity matrix
            K is node edge adjacency matrix
            d is constraints vector
    '''
    n_papers, n_reviewers = affinity_matrix.shape
    n_edges = np.count_nonzero(affinity_matrix)

    coo_affinity_matrix = sp.coo_matrix(affinity_matrix)

    N_p = sp.dok_matrix((n_papers, n_edges), dtype=np.float)
    N_p[coo_affinity_matrix.row, range(n_edges)] = 1

    N_r = sp.dok_matrix((n_reviewers, n_edges), dtype=np.float)
    N_r[coo_affinity_matrix.col, range(n_edges)] = 1

    K = sp.vstack([N_p, N_r, sp.identity(n_edges), -sp.identity(n_edges)])

    d = [max_reviewers_per_paper] * n_papers + [max_papers_per_reviewer] * n_reviewers + [1] * n_edges + [0] * n_edges
    d = np.atleast_2d(d).T

    return coo_affinity_matrix.data, K, d


def create_assignment(solution, affinity_matrix, threshold):
    '''
    Given a solution for paper assignments with affinity matrix, produce the actual assignment matrix.
    where   threshold is to determine whether one paper has connection with one reviewer or not
    '''
    n_papers, n_reviewers = affinity_matrix.shape
    coo_affinity_matrix = sp.coo_matrix(affinity_matrix)

    assign_edges = np.array(solution > threshold).ravel()

    assignment_matrix = np.zeros((n_papers, n_reviewers))
    assignment_matrix[coo_affinity_matrix.row[assign_edges], coo_affinity_matrix.col[assign_edges]] = 1

    return assignment_matrix
