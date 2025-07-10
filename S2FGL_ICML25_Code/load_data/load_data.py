from torch_geometric.datasets import Planetoid, WebKB, \
    HeterophilousGraphDataset
import torch
import numpy as np

def rand_train_test_idx(label, train_prop, valid_prop, test_prop, ignore_negative=True):
    labeled_nodes = torch.where(label != -1)[0]

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)
    test_num = int(n * test_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:train_num + valid_num + test_num]

    train_idx = train_indices
    valid_idx = val_indices
    test_idx = test_indices

    return {'train': train_idx.numpy(), 'valid': valid_idx.numpy(), 'test': test_idx.numpy()}

def index_to_mask(splits_lst, num_nodes):
    mask_len = len(splits_lst)
    train_mask = torch.zeros((mask_len, num_nodes), dtype=torch.bool)
    val_mask = torch.zeros((mask_len, num_nodes), dtype=torch.bool)
    test_mask = torch.zeros((mask_len, num_nodes), dtype=torch.bool)

    for i in range(mask_len):
        train_mask[i][splits_lst[i]['train']] = True
        val_mask[i][splits_lst[i]['valid']] = True
        test_mask[i][splits_lst[i]['test']] = True

    return train_mask.T, val_mask.T, test_mask.T

def load_dataset(dataname, train_prop, valid_prop, test_prop, num_masks):
    assert dataname in ('cora', 'citeseer', 'pubmed', 'texas', 'wisconsin', 'minesweeper'), 'Invalid dataset'
    if dataname in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='', name=dataname)
        data = dataset[0]
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)
    elif dataname in ['texas', 'wisconsin']:
        dataset = WebKB(root='', name=dataname)
        data = dataset[0]
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)
    elif dataname in ['minesweeper']:
        dataset = HeterophilousGraphDataset(root='dataset/', name=dataname)
        data = dataset[0]
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)
    return data










