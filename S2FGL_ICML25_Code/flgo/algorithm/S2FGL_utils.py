import numpy as np
import torch
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.data import Data

def personalized_pagerank(edge_index, num_nodes, alpha=0.85, eps=1e-6, device='cuda'):
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)
    degree = torch.sum(adj, dim=1)
    deg_inv = torch.diag(1.0 / (degree + eps))
    A_hat = deg_inv @ adj
    I = torch.eye(num_nodes, device=device)
    P = torch.inverse(I - (1 - alpha) * A_hat)
    return P

def structure_inertia_score(P, t):
    sis = torch.sum(torch.max(P * t, dim=1)[0])
    return sis

def compute_average_lps_test_nodes(data, alpha=0.85, device='cuda'):

    edge_index = data.edge_index.to(device)
    train_mask = data.train_mask.to(device)
    test_mask = data.test_mask.to(device)

    num_nodes = data.x.size(0)
    P = personalized_pagerank(edge_index, num_nodes, alpha, device=device)
    t = train_mask.clone().float().to(device)
    Pt = P @ t

    test_LPS = Pt[test_mask]


    if len(test_LPS) > 0:
        average_LPS = test_LPS.mean().item()
    else:
        average_LPS = 0.0

    return average_LPS


def compute_sis(data, alpha=0.85, device='cuda'):
    edge_index = data.edge_index.to(device)
    num_nodes = data.num_nodes
    train_mask = data.train_mask.to(device)

    P = personalized_pagerank(edge_index, num_nodes, alpha, device=device)

    t = train_mask.clone().float()

    sis = structure_inertia_score(P, t)

    return sis.item()

def personalized_pagerank(edge_index, num_nodes, alpha=0.85, eps=1e-6, device='cuda'):
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)

    degree = torch.sum(adj, dim=1)
    deg_inv = torch.diag(1.0 / (degree + eps))
    A_hat = deg_inv @ adj

    I = torch.eye(num_nodes, device=device)
    P = torch.inverse(I - (1 - alpha) * A_hat)

    return P


def structure_inertia_score(P, t):
    sis = torch.sum(torch.max(P * t, dim=1)[0])
    return sis

def select_important_nodes_lis(data, alpha=0.85, device='cuda'):
    edge_index = data.edge_index.to(device)
    num_nodes = data.num_nodes
    train_mask = data.train_mask.to(device)
    t = train_mask.float().to(device)
    P = personalized_pagerank(edge_index, num_nodes, alpha, device=device)
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)
    degree = torch.sum(adj, dim=1)
    deg_inv = torch.diag(1.0 / (degree + 1e-6)).to(device)
    A_hat = deg_inv @ adj
    I = torch.eye(num_nodes, device=device)
    P_prime = alpha * torch.inverse(I - (1 - alpha) * A_hat)
    Lambda_s = torch.max(P * t, dim=1).values
    labeled_nodes = torch.where(train_mask)[0].to(device)
    Lambda_l = torch.sum(P_prime[:, labeled_nodes], dim=1)
    Lambda_SALC = Lambda_s + Lambda_l
    top_k = int(num_nodes / 3)
    _, top_indices = torch.topk(Lambda_SALC, top_k)
    labels_of_important_nodes = data.y[top_indices].to(device)
    return top_indices, labels_of_important_nodes

def select_important_nodes_lis_K(data, alpha=0.85, device='cuda'):
    edge_index = data.edge_index.to(device)
    num_nodes = data.num_nodes
    train_mask = data.train_mask.to(device)
    t = train_mask.float().to(device)
    P = personalized_pagerank(edge_index, num_nodes, alpha, device=device)
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)
    degree = torch.sum(adj, dim=1)
    deg_inv = torch.diag(1.0 / (degree + 1e-6)).to(device)
    A_hat = deg_inv @ adj
    I = torch.eye(num_nodes, device=device)
    P_prime = alpha * torch.inverse(I - (1 - alpha) * A_hat)
    Lambda_s = torch.max(P * t, dim=1).values
    labeled_nodes = torch.where(train_mask)[0].to(device)
    Lambda_l = torch.sum(P_prime[:, labeled_nodes], dim=1)
    Lambda_SALC = Lambda_s + Lambda_l
    top_k = int(num_nodes/5)
    _, top_indices = torch.topk(Lambda_SALC, top_k)
    return top_indices
