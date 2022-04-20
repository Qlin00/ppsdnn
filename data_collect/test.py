import torch
from utils import get_features_of_sparse_matrix


A = torch.tensor([[0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 1., 0.],
        [0., 0., 1., 1., 1., 0.]])

print(A)
print(get_features_of_sparse_matrix(A, 1))