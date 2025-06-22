from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_scipy_sparse_matrix, degree
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian, add_self_loops
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from grakel import Graph
from grakel.kernels import MultiscaleLaplacian

def personalized_pagerank(edge_index, num_nodes, alpha=0.85, eps=1e-6, device='cuda'):
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)
    degree = torch.sum(adj, dim=1)
    deg_inv = torch.diag(1.0 / (degree + eps))
    A_hat = deg_inv @ adj
    I = torch.eye(num_nodes, device=device)
    P = torch.inverse(I - (1 - alpha) * A_hat)

    return P

def lps_loss(P, t):
    Pt = P @ t
    mean_Pt = Pt.mean()
    loss = torch.norm(Pt - mean_Pt, p=2) ** 2
    return loss

def structure_inertia_score(P, t):
    sis = torch.sum(torch.max(P * t, dim=1)[0])
    return sis

def select_important_nodes(data, alpha=0.85, lr=0.1, num_steps=300, beta=0.0001, gamma=0.1, K=None, device='cuda'):
    edge_index = data.edge_index.to(device)
    num_nodes = data.num_nodes
    labels = data.y.to(device)

    if K is None:
        K = num_nodes // 3
    P = personalized_pagerank(edge_index, num_nodes, alpha, device=device)

    t = torch.zeros(num_nodes, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([t], lr=lr)

    if hasattr(data, 'train_mask'):
        labels_mask = data.train_mask.float().to(device)
    else:
        labels_mask = (data.y != -1).float().to(device)

    for _ in range(num_steps):
        optimizer.zero_grad()

        loss_lps = lps_loss(P, t)

        sis = structure_inertia_score(P, t)

        label_term = gamma * ((1 - labels_mask) * t).sum()

        total_loss = loss_lps - beta * sis + label_term

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            t.clamp_(0, 1)

    important_nodes = torch.topk(t, K).indices

    representative_labels = torch.full((K,), -1, dtype=torch.long, device=device)

    for i, node_idx in enumerate(important_nodes):
        if labels_mask[node_idx]:
            representative_labels[i] = labels[node_idx]

    return important_nodes, representative_labels


def pyg_data_to_graph(pyg_data):

    edge_index = pyg_data.edge_index.cpu().numpy()
    num_nodes = pyg_data.num_nodes


    edges = []
    for src, dst in edge_index.T:
        edges.append((int(src), int(dst)))


    if pyg_data.x is not None:
        node_attributes = {}
        for i in range(num_nodes):
            node_attributes[i] = pyg_data.x[i].cpu().numpy()
    else:
        raise ValueError("Node attributes are required for the MultiscaleLaplacian kernel.")


    graph = {
        'edges': edges,
        'node_attributes': node_attributes
    }

    return graph

def plot_graph_similarity_heatmap(graph_data_list):

    graphs_dicts = [pyg_data_to_graph(data) for data in graph_data_list]


    graphs = [Graph(graph_dict, graph_format='dictionary') for graph_dict in graphs_dicts]


    ml_kernel = MultiscaleLaplacian()


    similarity_matrix = ml_kernel.fit_transform(graphs)


    print("Graph Similarity Matrix:\n", similarity_matrix)


    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', cbar=True)
    plt.title("Graph Similarity Heatmap")
    plt.xlabel("Graph Index")
    plt.ylabel("Graph Index")
    plt.show()

    return similarity_matrix

def compute_assortative_edge_ratios(data1, data2):

    edge_list1 = data1.edge_index.t().tolist()
    edge_list2 = data2.edge_index.t().tolist()

    edge_set1 = set(frozenset((u, v)) for u, v in edge_list1)
    edge_set2 = set(frozenset((u, v)) for u, v in edge_list2)

    broken_edges = edge_set1 - edge_set2

    y = data1.y

    def count_assortative_edges(edge_set):
        num_assortative = 0
        for edge in edge_set:
            u, v = tuple(edge)
            if y[u] == y[v]:
                num_assortative += 1
        return num_assortative


    num_assortative_broken = count_assortative_edges(broken_edges)
    num_broken_edges = len(broken_edges)

    num_assortative_original = count_assortative_edges(edge_set1)
    num_original_edges = len(edge_set1)

    if num_broken_edges == 0:
        broken_edge_assortative_ratio = 0.0
    else:
        broken_edge_assortative_ratio = num_assortative_broken / num_broken_edges

    if num_original_edges == 0:
        original_edge_assortative_ratio = 0.0
    else:
        original_edge_assortative_ratio = num_assortative_original / num_original_edges

    return broken_edge_assortative_ratio, original_edge_assortative_ratio


def cal_top_bottom_eigenvectors(data, num_eigenvalues=20, is_self_loop=False):

    def cal_eigenvalues_and_vectors(data, num_eigenvalues, is_self_loop, largest):
        edge_index, weight = data.edge_index, data.edge_weight
        num_nodes = data.x.size(0)
        if weight is None:
            weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)

        if is_self_loop:
            edge_index, weight = add_self_loops(edge_index, weight, num_nodes=num_nodes)

        deg = degree(edge_index[0], num_nodes, dtype=weight.dtype).to(edge_index.device)

        laplacian = torch.diag(deg) - torch.sparse_coo_tensor(edge_index, weight, (num_nodes, num_nodes)).to_dense()

        isolated_nodes = (deg == 0).nonzero(as_tuple=True)[0]
        laplacian[isolated_nodes, isolated_nodes] = 0

        laplacian = laplacian.cpu().numpy()

        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        if largest:
            selected_indices = np.argsort(eigenvalues)[-num_eigenvalues:]
        else:
            selected_indices = np.argsort(eigenvalues)[:num_eigenvalues]

        selected_eigenvectors = eigenvectors[:, selected_indices]

        return eigenvalues[selected_indices], selected_eigenvectors

    _, smallest_eigenvectors = cal_eigenvalues_and_vectors(data, num_eigenvalues, is_self_loop, largest=False)

    _, largest_eigenvectors = cal_eigenvalues_and_vectors(data, num_eigenvalues, is_self_loop, largest=True)

    return smallest_eigenvectors, largest_eigenvectors

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

def lps_loss(P, t):
    Pt = P @ t
    mean_Pt = Pt.mean()
    loss = torch.norm(Pt - mean_Pt, p=2) ** 2
    return loss

def merge_data_list(data_list):

    all_x = []
    all_edge_index = []
    all_y = []
    all_train_mask = []
    all_val_mask = []
    all_test_mask = []

    node_offset = 0

    for data in data_list:
        num_nodes = data.x.size(0)
        all_x.append(data.x)
        edge_index = data.edge_index + node_offset
        all_edge_index.append(edge_index)
        if data.y is not None:
            all_y.append(data.y)
        if data.train_mask is not None:
            all_train_mask.append(data.train_mask)
        if data.val_mask is not None:
            all_val_mask.append(data.val_mask)
        if data.test_mask is not None:
            all_test_mask.append(data.test_mask)
        node_offset += num_nodes

    merged_x = torch.cat(all_x, dim=0)
    merged_edge_index = torch.cat(all_edge_index, dim=1)
    merged_y = torch.cat(all_y, dim=0) if all_y else None

    merged_train_mask = torch.cat(all_train_mask, dim=0) if all_train_mask else None
    merged_val_mask = torch.cat(all_val_mask, dim=0) if all_val_mask else None
    merged_test_mask = torch.cat(all_test_mask, dim=0) if all_test_mask else None

    merged_data = Data(
        x=merged_x,
        edge_index=merged_edge_index,
        y=merged_y,
        train_mask=merged_train_mask,
        val_mask=merged_val_mask,
        test_mask=merged_test_mask
    )

    return merged_data


def select_important_nodes_latest(data, alpha=0.85, device='cuda', K=None):
    edge_index = data.edge_index.to(device)
    num_nodes = data.num_nodes
    labels = data.y.to(device)
    if K is None:
        K = num_nodes // 3
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        labeled_nodes = data.train_mask.nonzero(as_tuple=False).view(-1)
    else:
        labeled_nodes = (labels != -1).nonzero(as_tuple=False).view(-1)
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)
    degree = adj.sum(dim=1)
    deg_inv = torch.diag(1.0 / (degree + 1e-6))
    A_hat = deg_inv @ adj
    I = torch.eye(num_nodes, device=device)
    P = alpha * torch.inverse(I - (1 - alpha) * A_hat)

    PR = P.sum(dim=1)
    if labeled_nodes.numel() == 0:
        LIS = torch.zeros(num_nodes, device=device)
    else:
        P_labeled = P[labeled_nodes, :]
        LIS = P_labeled.sum(dim=0)

    LACS = PR * LIS
    important_nodes = torch.topk(LACS, K).indices
    representative_labels = torch.full((K,), -1, dtype=labels.dtype, device=device)
    selected_labels = labels[important_nodes]
    labeled_mask = selected_labels != -1
    representative_labels[labeled_mask] = selected_labels[labeled_mask]

    return important_nodes, representative_labels

def select_important_nodes_ori(data, alpha=0.5, lr=0.1, num_steps=300, lambda_reg=0.01, device='cuda'):
    edge_index = data.edge_index.to(device)
    num_nodes = data.x.size(0)
    train_mask = data.train_mask.to(device)
    labels = data.y.to(device)
    max_nodes = num_nodes // 3
    P = personalized_pagerank(edge_index, num_nodes, alpha, device=device)
    t = train_mask.clone().float().to(device)
    t.requires_grad = True

    optimizer = torch.optim.Adam([t], lr=lr)

    for _ in range(num_steps):
        optimizer.zero_grad()
        loss_lps = lps_loss(P, t)
        sis = structure_inertia_score(P, t)
        reg_term = lambda_reg * torch.norm(t, p=1)
        total_loss = loss_lps - 0.0001 * sis + reg_term
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            t.clamp_(0, 1)
            if torch.sum(t) > max_nodes:
                t *= max_nodes / torch.sum(t)
    important_nodes = torch.topk(t, max_nodes).indices
    representative_labels = torch.full((max_nodes,), -1, dtype=torch.long, device=device)

    for i, node_idx in enumerate(important_nodes):
        if train_mask[node_idx]:
            representative_labels[i] = labels[node_idx]

    return important_nodes, representative_labels

import torch
from torch_geometric.utils import to_dense_adj, add_self_loops

def compute_ppr(edge_index, num_nodes, alpha=0.85, device='cuda'):

    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)
    degree = adj.sum(dim=1)
    deg_inv = torch.diag(1.0 / (degree + 1e-6))
    A_hat = deg_inv @ adj
    I = torch.eye(num_nodes, device=device)

    P = alpha * torch.inverse(I - (1 - alpha) * A_hat)
    return P


def compute_LIS(P, labeled_nodes):

    if labeled_nodes.numel() == 0:
        return torch.zeros(P.size(0), device=P.device)
    P_labeled = P[labeled_nodes, :]
    LIS = P_labeled.sum(dim=0)
    return LIS


def compute_SIS_per_node(P, t):
    SIS, _ = torch.max(P * t.unsqueeze(0), dim=1)
    return SIS



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

    top_k = int(num_nodes/3)
    _, top_indices = torch.topk(Lambda_SALC, top_k)
    return top_indices