import torch
import numpy as np
import math


def get_features_of_sparse_matrix(sparse_matrix, block_size):
    """
    This is a function that get the features of sparse_matrix.
    The features: 
        n_row: number of rows
        n_col: number of cols
        nnz:   number of non-zeros elements
        density: the density of sparse matrix, nnz*(n_row*n_col)
        nnz_per_row_max: the max number of non-zeros element in any row
        nnz_per_row_min: the min number of non-zeros element in any row
        nnz_per_row_mean: the mean number of non-zeros element for all row
        nnz_blocks_per_row_max: the max number of blocks of Nw consecutive columns with at least one non-zero in any row
        nnz_blocks_per_row_min: the min number of blocks of Nw consecutive columns with at least one non-zero in any row
        nnz_blocks_per_row_mean: the mean number of blocks of Nw consecutive columns with at least one non-zero for all row
        
    
    Parameters:
    sparse_matrix - a 2D tensor of this sparse matrix, type:torch.tensor 
    block_size - this size of thread block, type:int
    
    Returns:
        a list of features:[n_row, n_col, nnz, density, nnz_per_row_max, nnz_per_row_min, nnz_per_row_mean,\
            nnz_blocks_per_row_max, nnz_blocks_per_row_min, nnz_blocks_per_row_mean]
    """
    sparse_matrix = sparse_matrix.numpy()
    n_row, n_col = sparse_matrix.shape
    # print(np.pad(sparse_matrix,(0, math.ceil(n_col/block_size)*block_size-n_col)).shape)
    nnz_per_row_list = np.count_nonzero(sparse_matrix, axis=1)

    nnz_blocks_per_row_list = np.count_nonzero(\
        np.count_nonzero(np.pad(sparse_matrix,((0, 0), (0, math.ceil(n_col/block_size)*block_size-n_col)))\
        .reshape(n_row, -1, block_size), axis = 2),\
        axis = 1
    )

    nnz = nnz_per_row_list.sum()
    density = nnz / (n_row * n_col)
    nnz_per_row_max = nnz_per_row_list.max()
    nnz_per_row_min = nnz_per_row_list.min()
    nnz_per_row_mean = nnz_per_row_list.mean()
    nnz_blocks_per_row_max = nnz_blocks_per_row_list.max()
    nnz_blocks_per_row_min = nnz_blocks_per_row_list.min()
    nnz_blocks_per_row_mean = nnz_blocks_per_row_list.mean()

    return [n_row, n_col, nnz, density, nnz_per_row_max, nnz_per_row_min, nnz_per_row_mean,\
            nnz_blocks_per_row_max, nnz_blocks_per_row_min, nnz_blocks_per_row_mean]



def get_latency_of_pruned_model(pruned_model = None): 
    """
    This is a function that get the latency of pruned_model.     
    In order to get accuracy result, this function perform warm up and multiply repeat run model.
    
    Parameters:
    model - a pruned model  
    
    Returns:
        the mean value of inference time for all times
    """
    x = torch.rand([32, 3, 64, 64])

    #warm_up
    for _ in range(20):
        _ = pruned_model(x)

    times = 100
    result = 0
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    
    with torch.no_grad():
        for _ in range(times):
            start.record()
            pruned_model(x)      
            end.record()
            torch.cuda.synchronize()
            l = start.elapsed_time(end)
            result += l

    result = result / times
    return result



def get_features_of_pruned_model(pruned_model=None):
    """
    convert the pruned_model to graphdata and save
    """
    pass