# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import random
import torch
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

MAX_attr = torch.tensor([1]*13)

def get_MAX_attr(attr):
    global MAX_attr
    comp = (MAX_attr > attr).int()
    temp = MAX_attr * comp
    MAX_attr = temp + attr*(1-comp)

class GNNDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, device="nvidia_2080ti", split_ratio=0.8, testing = False,test_file=None):
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
        self.nodename2id = {}
        self.id2nodename = {}
        self.op_types = set()
        self.opname2id = {}
        self.raw_data = {}
        self.name_list = []
        self.latencies = {}
        #self.data_dir = bench_dataset(data_folder=__user_dataset_folder__)
        self.data_dir = os.listdir("./dataset")
        self.load_model_archs_and_latencies(self.data_dir)
        self.construct_attrs()
        self.name_list = list(filter(lambda x: x in self.latencies, self.name_list))
        self.MAX_NORM = MAX_attr

    def load_model_archs_and_latencies(self, data_dir):
        for filename in data_dir:
            if self.testing == True and self.train==True and filename in self.test_file:
                print("training dataset skip ",filename)
                continue
            
            if self.testing == True and self.train==False and filename not in self.test_file:
                print("testint dataset skip ",filename)
                continue
            
            self.load_model(os.path.join("dataset",filename))

    def load_model(self, fpath):
        """
        Load a concrete model type.
        """
        assert os.path.exists(fpath), '{} does not exists'.format(fpath)
        print(fpath)
        with jsonlines.open(fpath) as reader:
            _names = []
            for obj in reader:
                if obj[self.device]:
                    _names.append(obj['id'])
                    self.latencies[obj['id']] = float(obj[self.device])

            _names = sorted(_names)
            split_ratio = self.split_ratio if self.train else 1-self.split_ratio
            count = int(len(_names) * split_ratio)

            if self.train:
                _model_names = _names[:count]
            else:
                _model_names = _names[-1*count:]

            self.name_list.extend(_model_names)

        with jsonlines.open(fpath) as reader:
            for obj in reader:
                if obj['id'] in _model_names:
                    model_name = obj['id']
                    model_data = obj['graph']
                    self.parse_model(model_name, model_data)
                    self.raw_data[model_name] = model_data
    
    def construct_attrs(self):
        """
        Construct the attributes matrix for each model.
        Attributes tensor:
        one-hot encoded type + input_channel , output_channel,
        input_h, input_w + kernel_size + stride
        """
        print("*******************************")
        print(self.op_types)
        print("*******************************")
        self.op_types = ('Gemm', 'MaxPool', 'Pad', 'Slice', 'Concat', 'Add', 'GlobalAveragePool', 'BatchNormalization', 'Reshape', 'AveragePool', 'Relu', 'Conv') 
        op_types_list = list(sorted(self.op_types))
        for i, _op in enumerate(op_types_list):
            self.opname2id[_op] = i
        n_op_type = len(self.op_types)
        attr_len = n_op_type + 13
        for model_name in self.raw_data:
            n_node = len(self.raw_data[model_name])
            # print("Model: ", model_name, " Number of Nodes: ", n_node)
            t_attr = torch.zeros(n_node, attr_len)
            for node in self.raw_data[model_name]:
                node_attr = self.raw_data[model_name][node]
                nid = self.nodename2id[model_name][node]
                op_type = node_attr['attr']['type']
                op_id = self.opname2id[op_type]
                t_attr[nid][op_id] = 1
                other_attrs = self.parse_node(model_name, node)
                t_attr[nid][-13:] = other_attrs
            self.attrs[model_name] = t_attr

    def parse_node(self, model_name, node_name):
        """
        Parse the attributes of specified node
        Get the input_c, output_c, input_h, input_w, kernel_size, stride
        of this node. Note: filled with 0 by default if this doesn't have
        coressponding attribute.
        [input channel, output channel, input height, input width, kernel size, stride,padding, output height, output width, input tensor size, output tensor size, FLOPs, param size]
        """
        node_data = self.raw_data[model_name][node_name]
        t_attr = torch.zeros(13)
        op_type = node_data['attr']['type']
        if op_type =='Conv2D' or op_type == 'Conv' :
            kernel_size = node_data['attr']['attr']['kernel_shape'][0]
            padding = node_data['attr']['attr']['pads'][0]
            stride = node_data['attr']['attr']['strides'][0]
            _, hout, wout, out_c = node_data['attr']['output_shape'][0]
            _, hin, win, in_c = node_data['attr']['input_shape'][0]
            input_tensor_size = 128*in_c * hin * win
            output_tensor_size = 128*out_c * hout * wout
            FLOPs = out_c * hout * wout * (in_c * kernel_size * kernel_size +1) 
            param_size = in_c * out_c * kernel_size * kernel_size + 1 
            t_attr = torch.tensor([in_c, out_c, hin, win, kernel_size, stride, padding, hout, wout, input_tensor_size, output_tensor_size, FLOPs, param_size])
        elif op_type == 'DepthwiseConv2dNative': #我目前的NN没有这个算子
            weight_shape = node_data['attr']['attr']['weight_shape']
            kernel_size, _, in_c, out_c = weight_shape
            stride, _= node_data['attr']['attr']['strides']
            padding,_,_, _= node_data['attr']['attr']['pads']
            _, hout, wout, _ = node_data['attr']['output_shape'][0]
            _, hin, win, _ = node_data['attr']['input_shape'][0]
            input_tensor_size = 128*in_c * hin * win
            output_tensor_size = 128*out_c * hout * wout
            FLOPs = out_c * hout * wout * (in_c * kernel_size * kernel_size +1) 
            param_size = in_c * out_c * kernel_size * kernel_size + 1 
            t_attr = torch.tensor([in_c, out_c, hin, win, kernel_size, stride, padding, hout, wout, input_tensor_size, output_tensor_size, FLOPs, param_size])
        elif op_type == 'MatMul': #目前没有这个算子
            in_node = node_data['inbounds'][0]
            in_shape = self.raw_data[model_name][in_node]['attr']['output_shape'][0]
            in_c = in_shape[-1]
            out_c = node_data['attr']['output_shape'][0][-1]
            t_attr[0] = in_c
            t_attr[1] = out_c
        elif op_type == 'Relu':
            kernel_size = 0
            stride = 0
            padding = 0
            if len(node_data['attr']['output_shape'][0])==4:
                _, hout, wout, out_c = node_data['attr']['output_shape'][0]
                _, hin, win, in_c = node_data['attr']['input_shape'][0]
            else:
                hout, wout = node_data['attr']['output_shape'][0]
                hin, win = node_data['attr']['input_shape'][0]
                out_c = 1
                in_c = 1

            input_tensor_size = 128*in_c * hin * win
            output_tensor_size = 128*out_c * hout * wout
            FLOPs = output_tensor_size 
            param_size = 0 
            t_attr = torch.tensor([in_c, out_c, hin, win, kernel_size, stride, padding, hout, wout, input_tensor_size, output_tensor_size, FLOPs, param_size])
        elif op_type == 'BatchNormalization':
            kernel_size = 0
            stride = 0
            padding = 0
            _, hout, wout, out_c = node_data['attr']['output_shape'][0]
            _, hin, win, in_c = node_data['attr']['input_shape'][0]
            input_tensor_size = 128*in_c * hin * win
            output_tensor_size = 128*out_c * hout * wout
            FLOPs = input_tensor_size * 2 
            param_size = out_c * 4 
            t_attr = torch.tensor([in_c, out_c, hin, win, kernel_size, stride, padding, hout, wout, input_tensor_size, output_tensor_size, FLOPs, param_size])
        elif op_type == 'MaxPool' or op_type == 'AveragePool' or op_type == 'GlobalAveragePool':
            kernel_size = 0
            stride = 0
            padding = 0
            _, hout, wout, out_c = node_data['attr']['output_shape'][0]
            _, hin, win, in_c = node_data['attr']['input_shape'][0]
            input_tensor_size = 128*in_c * hin * win
            output_tensor_size = 128*out_c * hout * wout
            FLOPs = input_tensor_size  
            param_size = 0 
            t_attr = torch.tensor([in_c, out_c, hin, win, kernel_size, stride, padding, hout, wout, input_tensor_size, output_tensor_size, FLOPs, param_size])
        elif op_type == 'Gemm':
            kernel_size = 0
            stride = 0
            padding = 0
            in_c = 0
            out_c = 0
            hout, wout = node_data['attr']['output_shape'][0]
            hin, win = node_data['attr']['input_shape'][0]
            input_tensor_size = 128*hin * win
            output_tensor_size = 128*hout * wout
            FLOPs = 2 * win *wout 
            param_size = win*wout + 1 
            t_attr = torch.tensor([in_c, out_c, hin, win, kernel_size, stride, padding, hout, wout, input_tensor_size, output_tensor_size, FLOPs, param_size])
        elif op_type == 'Pad':
            kernel_size = 0
            stride = 0
            padding = 0
            _, hout, wout, out_c = node_data['attr']['output_shape'][0]
            _, hin, win, in_c = node_data['attr']['input_shape'][0]
            input_tensor_size = 128*in_c * hin * win
            output_tensor_size = 128*out_c * hout * wout
            FLOPs = 0 
            param_size = 0 
            t_attr = torch.tensor([in_c, out_c, hin, win, kernel_size, stride, padding, hout, wout, input_tensor_size, output_tensor_size, FLOPs, param_size])
        elif op_type == 'Add':
            kernel_size = 0
            stride = 0
            padding = 0
            _, hout, wout, out_c = node_data['attr']['output_shape'][0]
            _, hin, win, in_c = node_data['attr']['input_shape'][0]
            input_tensor_size = 128*in_c * hin * win * 2
            output_tensor_size = 128*out_c * hout * wout
            FLOPs = output_tensor_size
            param_size = 0 
            t_attr = torch.tensor([in_c, out_c, hin, win, kernel_size, stride, padding, hout, wout, input_tensor_size, output_tensor_size, FLOPs, param_size])
        elif op_type == 'Concat':
            kernel_size = 0
            stride = 0
            padding = 0
            _, hout, wout, out_c = node_data['attr']['output_shape'][0]
            _, hin, win, _ = node_data['attr']['input_shape'][0]
            in_c = out_c
            input_tensor_size = 128*out_c * hout * wout
            output_tensor_size = 128*out_c * hout * wout
            FLOPs = 0
            param_size = 0 
            t_attr = torch.tensor([in_c, out_c, hin, win, kernel_size, stride, padding, hout, wout, input_tensor_size, output_tensor_size, FLOPs, param_size])
        elif op_type == 'Slice':
            kernel_size = 0
            stride = 0
            padding = 0
            _, hout, wout, out_c = node_data['attr']['output_shape'][0]
            _, hin, win, in_c = node_data['attr']['input_shape'][0]
            input_tensor_size = 128*in_c * hin * win
            output_tensor_size = 128*out_c * hout * wout
            FLOPs = 0
            param_size = 0 
            t_attr = torch.tensor([in_c, out_c, hin, win, kernel_size, stride, padding, hout, wout, input_tensor_size, output_tensor_size, FLOPs, param_size])
        elif op_type == 'Reshape':
            kernel_size = 0
            stride = 0
            padding = 0
            hout, wout  = node_data['attr']['output_shape'][0]
            _, hin, win, in_c = node_data['attr']['input_shape'][0]
            out_c = 1
            input_tensor_size = 128*in_c * hin * win
            output_tensor_size = 128*out_c * hout * wout
            FLOPs = 0
            param_size = 0 
            t_attr = torch.tensor([in_c, out_c, hin, win, kernel_size, stride, padding, hout, wout, input_tensor_size, output_tensor_size, FLOPs, param_size])
            
        elif len(node_data['inbounds']):
            in_node = node_data['inbounds'][0]
            h, w, in_c, out_c = 0, 0, 0, 0
            in_shape = self.raw_data[model_name][in_node]['attr']['output_shape'][0]
            in_c = in_shape[-1]
            if 'ConCat' in op_type:
                for i in range(1, len(node_data['in_bounds'])):
                    in_shape = self.raw_data[node_data['in_bounds']
                                             [i]]['attr']['output_shape'][0]
                    in_c += in_shape[-1]
            if len(node_data['attr']['output_shape']):
                out_shape = node_data['attr']['output_shape'][0]
                # N, H, W, C
                out_c = out_shape[-1]
                if len(out_shape) == 4:
                    h, w = out_shape[1], out_shape[2]
            t_attr[0:4] = torch.tensor([in_c, out_c, h, w])
         
        get_MAX_attr(t_attr)
        return t_attr

    def parse_model(self, model_name, model_data):
        """
        Parse the model data and build the adjacent matrixes
        """
        n_nodes = len(model_data)
        m_adj = torch.zeros(n_nodes, n_nodes, dtype=torch.int32)
        id2name = {}
        name2id = {}
        tmp_node_id = 0
        # build the mapping between the node name and node id

        for node_name in model_data.keys():
            id2name[tmp_node_id] = node_name
            name2id[node_name] = tmp_node_id
            op_type = model_data[node_name]['attr']['type']
            self.op_types.add(op_type)
            tmp_node_id += 1

        for node_name in model_data:
            cur_id = name2id[node_name]
            for node in model_data[node_name]['inbounds']:
                if node not in name2id:
                    # weight node
                    continue
                in_id = name2id[node]
                m_adj[in_id][cur_id] = 1
            for node in model_data[node_name]['outbounds']:
                if node not in name2id:
                    # weight node
                    continue
                out_id = name2id[node]
                m_adj[cur_id][out_id] = 1
        
        for idx in range(n_nodes):
            m_adj[idx][idx] = 1

        self.adjs[model_name] = m_adj
        self.nodename2id[model_name] = name2id
        self.id2nodename[model_name] = id2name

    def __getitem__(self, index):
        model_name = self.name_list[index]
        return (self.adjs[model_name], self.attrs[model_name]), self.latencies[model_name], self.op_types

    def __len__(self):
        return len(self.name_list)


class GNNDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset, shuffle=False, batchsize=1):
        self.dataset = dataset
        self.op_num = len(dataset.op_types)
        self.shuffle = shuffle
        self.batchsize = batchsize
        self.length = len(self.dataset)
        self.indexes = list(range(self.length))
        self.pos = 0
        self.graphs = {}
        self.latencies = {}
        self.construct_graphs()

    def construct_graphs(self):
        dgl = try_import_dgl()
        MAXNORM = torch.tensor([1]*12+self.dataset.MAX_NORM.tolist())
        for gid in range(self.length):
            (adj, attrs), latency, op_types = self.dataset[gid]
            u, v = torch.nonzero(adj, as_tuple=True)
            graph = dgl.graph((u, v))
            #MAX_NORM = torch.tensor([1]*len(op_types) + [6963, 6963, 224, 224, 11, 4]) # 范数
            attrs = attrs / MAXNORM
            graph.ndata['h'] = attrs
            self.graphs[gid] = graph
            self.latencies[gid] = latency

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
        return torch.tensor(batch_latencies), dgl.batch(batch_graphs)
