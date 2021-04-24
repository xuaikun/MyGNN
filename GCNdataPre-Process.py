# fork：https://github.com/NIRVANALAN/gcn_analysis/blob/master/notebook/Plantenoid%20Citation%20Data%20Format%20Transformation.ipynb
# 针对GCN-Tensorflow版输入数据格式与GCN-Pytorch版输入数据不同的问题
# 需要对原始数据进行更改，以生成.x,.y,.allx,.ally,.tx,.ty,.graph,.index,等数据等数据，代码如下：
import time
import argparse
from collections import defaultdict

import numpy as np
import pdb

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

# raw data version
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

path="G:/graduatestudy/coding/pytorch/gcn/data/cora/"
dataset="cora"
idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),dtype=np.dtype(str))
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
# features = normalize(features) # no normalization in plantoid

labels = encode_onehot(idx_features_labels[:, -1])
# build graph
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                dtype=np.int32)
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                 dtype=np.int32).reshape(edges_unordered.shape)
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(labels.shape[0], labels.shape[0]),
                    dtype=np.float32)

# build symmetric adjacency matrix
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
idx_train = range(140)
idx_val = range(200, 500)
idx_test = range(500, 1500)

"""
Loads input data from gcn/data directory

ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
    (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
    object;
ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

All objects above must be saved using python pickle module.

:param dataset_str: Dataset name
:return: All data input files loaded (as well the training/test data).
"""

save_root = "G:/graduatestudy/coding/pytorch/gcn/data"
import pickle
pickle.dump(features[idx_train], open(f"{save_root}/ind.cora.x", "wb"))
pickle.dump(sp.vstack((features[:idx_test[0]], features[idx_test[-1]+1:])), open(f"{save_root}/ind.cora.allx", "wb" ))
pickle.dump(features[idx_test], open(f"{save_root}/ind.cora.tx", "wb"))

pickle.dump(labels[idx_train], open(f"{save_root}/ind.cora.y", "wb"))
pickle.dump(labels[idx_test], open(f"{save_root}/ind.cora.ty", "wb"))
pickle.dump(np.vstack((labels[:idx_test[0]],labels[idx_test[-1]+1:])), open(f"{save_root}/ind.cora.ally", "wb"))

with open(f"{save_root}/ind.cora.test.index", "w") as f:
    for item in list(idx_test):
        f.write("%s\n" % item)

# ori_graph
array_adj = np.argwhere(adj.toarray())
ori_graph = defaultdict(list)
for edge in array_adj:
    ori_graph[edge[0]].append(edge[1])
pickle.dump(ori_graph, open(f"{save_root}/ind.cora.graph", "wb"))
