# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import random
import torch
import numpy as np
from scipy import sparse
import jsonlines


def try_import_dgl():
    try:
        import dgl
        return dgl
    except ImportError:
        print("You hava not install the dgl package")

__user_dataset_folder__ = 'dataset'


hws = [
    "nvidia_2080ti",
]
attr_len = 3
MAX_attr = torch.tensor([1]*attr_len)

def get_MAX_attr(attr):
    global MAX_attr
    comp = (MAX_attr > attr).int()
    temp = MAX_attr * comp
    MAX_attr = temp + attr*(1-comp)

class GNNDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, device="nvidia_2080ti", split_ratio=0.7, testing = False,test_file=None):
        """
        Dataloader of the Latency Dataset

        Parameters
        ----------
        data_dir : string
            Path to save the downloaded dataset
        train: bool
            Get the train dataset or the test dataset
        device: string
            The Device type of the corresponding latency
        shuffle: bool
            If shuffle the dataset at the begining of an epoch
        batch_size: int
            Batch size.
        split_ratio: float
            The ratio to split the train dataset and the test dataset.
        """
        err_str = "Not supported device type"
        assert device in hws, err_str
        self.testing = testing
        self.test_file = test_file
        self.device = device
        self.train = train
        self.split_ratio = split_ratio
        self.adjs = {}
        self.attrs = {}

        self.name_list = []
        self.latencies = {}
        self.Ms = {}
        #self.data_dir = bench_dataset(data_folder=__user_dataset_folder__)
        self.data_dir = "/data1/sgk_files/profiling/SMXPath.txt"
        self.fpath_l = '/home/linjunqing/sparsednn/result_cusparse2.csv'
        self.load_model_archs_and_latencies(self.data_dir, self.fpath_l)
        self.name_list = list(filter(lambda x: x in self.latencies, self.name_list))
        self.MAX_NORM = MAX_attr

    def load_model_archs_and_latencies(self, data_dir, fpath_l):
        with open(fpath_l) as f:
            f.readline()
            for line in f.readlines():
                arr = line.strip().split(',')
                self.latencies[arr[1]] = float(arr[-1])
            
        self.load_matrix(os.path.join("/data1/sgk_files/profiling",data_dir))

    def load_matrix(self, fpath):
        """
        Load a concrete matrix type.
        """
    
        with open(fpath) as f:
            _names = []
            for name in f.readlines():
                name = os.path.join("/data1/sgk_files/profiling",name.strip())
                # print(name)
                if name in self.latencies.keys():
                    _names.append(name)

            split_ratio = self.split_ratio if self.train else 1-self.split_ratio
            count = int(len(_names) * split_ratio)
            if self.train:
                _matrix_names = _names[:count]
                _matrix_names = _names[:7000]
            else:
                _matrix_names = _names[-1*count:]
                _matrix_names = _names[7000:10000]
            
            self.name_list.extend(_matrix_names)
            nums = 0
            for matrix_name in _matrix_names:
                N,K,M,sparse_ratio,rows, cols = self.parse_matrix_from_densefile(matrix_name)
                self.parse_matrix(N, K, matrix_name, rows, cols)
                self.Ms[matrix_name] = [sparse_ratio, M]

                nums += 1
                if nums % 1000 == 0:
                    print(nums)
                    
    
    def parse_matrix(self, N, K, matrix_name, rows, cols):
        """
        Parse the matrix data and build the adjacent matrixes
        """
        n_nodes = max(N, K)
        pos = np.arange(0, n_nodes)
        rows = np.concatenate((rows, pos),axis=0)
        cols = np.concatenate((cols, pos),axis=0)
        indegree = np.bincount(rows, minlength=n_nodes)
        outdegree = np.bincount(cols, minlength=n_nodes)
        # Ms = np.full((n_nodes,), fill_value=M)
        t_attr = torch.tensor(np.array([indegree,outdegree, pos])).transpose(0,1)
        get_MAX_attr(t_attr.max(0).values)

        self.adjs[matrix_name] = (rows, cols)

        self.attrs[matrix_name] = t_attr


    def parse_matrix_from_densefile(self, filename):
        # print(filename)
        with open(filename, 'r', encoding='utf-8') as f:
            matrixInfo = f.readline().split(",")
            N, K, M = int(matrixInfo[0]),int(matrixInfo[1]),int(matrixInfo[2])
            result  = ""
            for line in f.readlines():
                line = line.replace('[', '  ')
                line = line.replace(']', '  ')
                line = line.replace('\n', ' ')
                result += line
            matrix = np.fromstring(result, dtype=np.float32, sep=' ')
            matrix = matrix.reshape((N, K))
            # print(matrix)
            sparse_matrix = sparse.coo_matrix(matrix)
            sparse_ratio = sparse_matrix.nnz / (N*K)
            return N,K,M,sparse_ratio,sparse_matrix.row, sparse_matrix.col



    def __getitem__(self, index):
        matrix_name = self.name_list[index]
        return (self.adjs[matrix_name], self.attrs[matrix_name]), self.latencies[matrix_name], self.Ms[matrix_name]

    def __len__(self):
        return len(self.name_list)


class GNNDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset, shuffle=False, batchsize=1):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batchsize = batchsize
        self.length = len(self.dataset)
        self.indexes = list(range(self.length))
        self.pos = 0
        self.graphs = {}
        self.latencies = {}
        self.matrix_features = {}
        self.construct_graphs()

    def construct_graphs(self):
        dgl = try_import_dgl()
        MAXNORM = torch.tensor(self.dataset.MAX_NORM.tolist())
        for gid in range(self.length):
            (adj, attrs), latency, matrix_feature = self.dataset[gid]
            u, v = adj
            graph = dgl.graph((u, v))
            #MAX_NORM = torch.tensor([1]*len(op_types) + [6963, 6963, 224, 224, 11, 4]) # 范数
            attrs = attrs / MAXNORM
            graph.ndata['h'] = attrs
            self.graphs[gid] = graph
            self.latencies[gid] = latency
            self.matrix_features[gid] = matrix_feature

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indexes)
        self.pos = 0
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        dgl = try_import_dgl()
        start = self.pos
        end = min(start + self.batchsize, self.length)
        self.pos = end
        if end - start <= 0:
            raise StopIteration
        batch_indexes = self.indexes[start:end]
        batch_graphs = [self.graphs[i] for i in batch_indexes]
        batch_latencies = [self.latencies[i] for i in batch_indexes]
        batch_matrix_features = [self.matrix_features[i] for i in batch_indexes]
        return torch.tensor(batch_latencies), dgl.batch(batch_graphs), torch.tensor(batch_matrix_features)
