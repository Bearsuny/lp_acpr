import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ['create_assignment', 'format_assignment']


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


def format_assignment(assignment_matrix, src_label, dst_label, save_path):
    am_row, am_col = assignment_matrix.shape
    src_elements = np.array([i+1 for i in range(am_row)])
    dst_elements = np.array([i+1 for i in range(am_col)])
    output = [dst_elements[np.nonzero(assignment_matrix[i])[0]] for i in range(am_row)]

    src_pd = pd.Series(src_elements, name=src_label)
    dst_pd = pd.Series(output, name=dst_label)

    save_pd = pd.DataFrame({src_label: src_pd, dst_label: dst_pd})
    save_pd.to_csv(save_path, index=False, sep=' ')

    print(save_pd)
