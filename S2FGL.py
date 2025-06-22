import copy

import scipy.sparse as sp

from scipy.sparse.linalg import eigsh
from torch_geometric.utils import add_self_loops

from .fedbase import BasicServer
from .fedbase import BasicParty
from flgo.utils import fmodule
import torch
import collections
import flgo.benchmark
import math
import numpy as np
import random
import dgl
import torch.nn.functional as F
import torch.nn as nn

import torch




def compute_laplacian(similarity_matrix, reg=1e-5):
    degree_matrix = torch.diag(similarity_matrix.sum(dim=1))
    laplacian_matrix = degree_matrix - similarity_matrix
    laplacian_matrix += reg * torch.eye(laplacian_matrix.size(0), device=laplacian_matrix.device)  # 正则化
    return laplacian_matrix

def calculate_new_loss(local_features, global_features, top_k=10):
    def compute_sparse_similarity_matrix(features, k=3):
        device = features.device
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)

        values, indices = similarity_matrix.topk(k + 1, dim=1)
        num_nodes = features.size(0)

        row_indices = torch.arange(num_nodes, device=device).repeat_interleave(k + 1)
        col_indices = indices.view(-1)
        values = values.view(-1)
        sparse_matrix = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices]),
            values,
            size=(num_nodes, num_nodes)
        )
        return sparse_matrix

    def compute_sparse_laplacian(sparse_similarity_matrix):
        device = sparse_similarity_matrix.device

        degree_values = torch.sparse.sum(sparse_similarity_matrix, dim=1).to_dense()
        degree_matrix = torch.sparse_coo_tensor(
            torch.stack([torch.arange(degree_values.size(0), device=device),
                         torch.arange(degree_values.size(0), device=device)]),
            degree_values.to(device),
            size=sparse_similarity_matrix.shape,
            device=device
        )

        laplacian_matrix = degree_matrix - sparse_similarity_matrix
        return laplacian_matrix



    def top_bottom_eigenvectors_sparse(laplacian_matrix, k=10):
        laplacian_sparse = sp.coo_matrix(laplacian_matrix.to_dense().detach().cpu().numpy())
        eigenvalues_small, eigenvectors_small = eigsh(laplacian_sparse, k=k, which='SM')
        eigenvalues_large, eigenvectors_large = eigsh(laplacian_sparse, k=k, which='LM')

        top_k_smallest = torch.tensor(eigenvectors_small, dtype=torch.float32, device=laplacian_matrix.device)
        top_k_largest = torch.tensor(eigenvectors_large, dtype=torch.float32, device=laplacian_matrix.device)
        return top_k_smallest, top_k_largest

    def project_features_onto_eigenvectors(eigenvectors, features):
        projected_features = [torch.mm(torch.mm(eigenvectors[:, i].view(-1, 1), eigenvectors[:, i].view(1, -1)), features)
                              for i in range(eigenvectors.shape[1])]
        return projected_features

    def calculate_projection_loss(projected_local, projected_global):
        return sum(F.mse_loss(loc, glob) for loc, glob in zip(projected_local, projected_global))

    local_similarity_matrix = compute_sparse_similarity_matrix(local_features)
    global_similarity_matrix = compute_sparse_similarity_matrix(global_features)

    local_laplacian = compute_sparse_laplacian(local_similarity_matrix)
    global_laplacian = compute_sparse_laplacian(global_similarity_matrix)

    local_smallest_eigenvectors, local_largest_eigenvectors = top_bottom_eigenvectors_sparse(local_laplacian, k=top_k)
    global_smallest_eigenvectors, global_largest_eigenvectors =  top_bottom_eigenvectors_sparse(global_laplacian, k=top_k)

    local_proj_small = project_features_onto_eigenvectors(local_smallest_eigenvectors, local_features)
    global_proj_small = project_features_onto_eigenvectors(global_smallest_eigenvectors, global_features)
    local_proj_large = project_features_onto_eigenvectors(local_largest_eigenvectors, local_features)
    global_proj_large = project_features_onto_eigenvectors(global_largest_eigenvectors, global_features)

    loss_small = calculate_projection_loss(local_proj_small, global_proj_small)
    loss_large = calculate_projection_loss(local_proj_large, global_proj_large)

    total_loss = loss_small + loss_large
    return total_loss







def calculate_projection_loss(smallest_eigenvectors, largest_eigenvectors, output1, output2):
    def transform_output(eigenvectors_np, output_np):
        eigenvectors = torch.from_numpy(eigenvectors_np).float().to('cuda')
        output = output_np.float().to('cuda')

        transformed_outputs = []

        for i in range(eigenvectors.shape[1]):
            eigenvector = eigenvectors[:, i].view(-1, 1)
            projection_matrix = torch.mm(eigenvector, eigenvector.t())
            transformed_output = torch.mm(projection_matrix, output)
            transformed_outputs.append(transformed_output)
        return transformed_outputs
    proj_output1_small = transform_output(smallest_eigenvectors, output1)
    proj_output2_small = transform_output(smallest_eigenvectors, output2)
    proj_output1_large = transform_output(largest_eigenvectors, output1)
    proj_output2_large = transform_output(largest_eigenvectors, output2)
    loss_small = 0
    for o1, o2 in zip(proj_output1_small, proj_output2_small):
        loss_small += F.mse_loss(o1, o2)
    loss_large = 0
    for o1, o2 in zip(proj_output1_large, proj_output2_large):
        loss_large += F.mse_loss(o1, o2)
    return loss_large, loss_small



def calculate_weights(lps_scores):
    min_score = min(lps_scores)
    max_score = max(lps_scores)

    if max_score == min_score:
        return [1.0 / len(lps_scores)] * len(lps_scores)

    normalized_scores = [(score - min_score) / (max_score - min_score) for score in lps_scores]

    total = sum(normalized_scores)
    weights = [score / total for score in normalized_scores]

    return weights


def federated_knowledge_distillation_loss(local_features, global_features, codebook_embeddings, temperature=1, lamb=10):
    if isinstance(codebook_embeddings, np.ndarray):
        codebook_embeddings = codebook_embeddings.reshape(-1, codebook_embeddings.shape[-1])
        codebook_embeddings = torch.tensor(codebook_embeddings, dtype=local_features.dtype, device=local_features.device)
    tea_sim = F.cosine_similarity(global_features.unsqueeze(1), codebook_embeddings.unsqueeze(0), dim=-1)
    tea_soft_token_assignments = tea_sim / temperature
    tea_soft_token_assignments = F.softmax(tea_soft_token_assignments, dim=-1)
    stu_sim = F.cosine_similarity(local_features.unsqueeze(1), codebook_embeddings.unsqueeze(0), dim=-1)
    stu_soft_token_assignments = stu_sim / temperature
    stu_soft_token_assignments = F.log_softmax(stu_soft_token_assignments, dim=-1)

    kl_loss = F.kl_div(stu_soft_token_assignments, tea_soft_token_assignments, reduction='batchmean')
    kl_loss = kl_loss * (temperature ** 2) * lamb
    return kl_loss


def compute_prototypes(node_features, important_nodes, representative_labels, num_classes):

    prototypes = []
    sample_counts = []


    for class_id in range(num_classes):

        class_mask = (representative_labels == class_id)
        class_indices = important_nodes[class_mask]


        if len(class_indices) == 0:
            prototypes.append(torch.zeros(node_features.size(1), device=node_features.device))
            sample_counts.append(0)
        else:

            class_features = node_features[class_indices]

            prototype = class_features.mean(dim=0)

            prototypes.append(prototype)
            sample_counts.append(class_features.size(0))

    return prototypes, sample_counts

class Server(BasicServer):
    def check_nan_layers(self, models):

        for idx, model in enumerate(models):
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    self.gv['logger']['info'](f"Model {idx} layer '{name}' contains NaN values.")

    def aggregate(self, models: list, *args, **kwargs):

        if hasattr(self.clients[0], 'node_features'):

            codebook = None

            for k in range(4):
                global_prototypes = {}
                global_vols = {}


                sampled_clients = random.sample(self.clients, int(len(self.clients) * 0.7))


                num_classes = torch.max(self.clients[0].important_labels).item() + 1
                feature_dim = self.clients[0].node_features.size(1)

                for client in sampled_clients:

                    client.prototypes, client.proto_vols = compute_prototypes(
                        client.node_features,
                        client.important_nodes,
                        client.important_labels,
                        num_classes
                    )


                    if codebook is None:
                        codebook = np.zeros((num_classes, 4, feature_dim))


                    for i in range(num_classes):
                        proto = client.prototypes[i]
                        vol = client.proto_vols[i]

                        if i in global_prototypes:
                            global_prototypes[i] += proto * vol
                            global_vols[i] += vol
                        else:
                            global_prototypes[i] = proto * vol
                            global_vols[i] = vol


                for i in range(num_classes):
                    if global_vols[i] > 0:
                        codebook[i, k, :] = (global_prototypes[i] / global_vols[i]).cpu().detach().numpy()


            for client in self.clients:
                client.codebook = codebook
        if len(models) == 0:
            return self.model

        nan_exists = [m.has_nan() for m in models]


        if any(nan_exists):
            if all(nan_exists):
                raise ValueError("All the received local models have parameters of nan value.")

            self.gv.logger.info(
                'Warning("There exists nan-value in local models, which will be automatically removed from the aggregation list.")')

            new_models = []
            received_clients = []

            for ni, mi, cid in zip(nan_exists, models, self.received_clients):
                if ni: continue
                new_models.append(mi)
                received_clients.append(cid)
            self.received_clients = received_clients
            models = new_models


        lps_scores = [client.LPS for client in self.clients]
        lps_weights = calculate_weights(lps_scores)
        weights = lps_weights

        local_data_vols = [c.datavol for c in self.clients]
        total_data_vol = sum(local_data_vols)

        aggregated_params = collections.defaultdict(lambda: 0)


        for i, model in enumerate(models):
            for name, param in model.named_parameters():
                aggregated_params[name] += param.data * local_data_vols[i] / total_data_vol
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_params:
                    param.data.copy_(aggregated_params[name])

        return self.model



class Client(BasicParty):
    TaskCalculator = flgo.benchmark.base.BasicTaskCalculator
    def __init__(self, option={}):
        super().__init__()
        self.id = None
        # create local_movielens_recommendation dataset
        self.data_loader = None
        self.test_data = None
        self.val_data = None
        self.train_data = None
        self.model = None

        self.device = self.gv.apply_for_device()
        self.calculator = self.TaskCalculator(self.device, option['optimizer'])
        self._train_loader = None

        self.optimizer_name = option['optimizer']
        self.learning_rate = option['learning_rate']
        self.momentum = option['momentum']
        self.weight_decay = option['weight_decay']
        self.batch_size = option['batch_size']
        self.num_steps = option['num_steps']
        self.num_epochs = option['num_epochs']
        self.clip_grad = option['clip_grad']
        self.model = None
        self.test_batch_size = option['test_batch_size']
        self.loader_num_workers = option['num_workers']
        self.current_steps = 0

        self._effective_num_steps = self.num_steps
        self._latency = 0

        self.server = None

        self.option = option
        self.actions = {0: self.reply}
        self.default_action = self.reply

        self.model_g = None

        self.fr_low_g_list = None
        self.fr_high_g_list = None
        self.mlp_list = None

    @fmodule.with_multi_gpus
    def train(self, model):

        model.train()

        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        if self.model_g is not None:
            for param in self.model_g.parameters():
                param.requires_grad = False

        self.fr_low_g_list = []
        self.fr_high_g_list = []
        self.mlp_list = []


        for iter in range(self.num_steps):
            if iter == 0:
                '''for gcn_layer in model.model.gcns:
                    fr_low_g = gcn_layer.att_vec_low_unk
                    self.fr_low_g_list.append(fr_low_g)

                    fr_high_g = gcn_layer.att_vec_high_unk
                    self.fr_high_g_list.append(fr_high_g)

                    mlp_g = gcn_layer.att_vec_mlp
                    self.mlp_list.append(mlp_g)'''

            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()

            loss_dict, node_features, node_features_g = self.calculator.compute_loss(model, self.model_g, batch_data, self.adj_low_un,
                                                                    self.adj_low, self.adj_high, self.fr_low_g_list,
                                                                    self.fr_high_g_list, self.mlp_list)
            loss = loss_dict['loss']


            sub_features = node_features[self.important_nodes_k]


            sub_g = node_features_g[self.important_nodes_k]

            self.node_features = node_features

            if hasattr(self, 'codebook'):

                NLIR = federated_knowledge_distillation_loss(node_features, node_features_g, self.codebook, temperature=1, lamb = 60)

                FGMA = calculate_new_loss(sub_features, sub_g, 10)

                loss += NLIR

                loss += FGMA

                num_nodes = node_features.shape[0]

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
        return

    @fmodule.with_multi_gpus
    def test(self, model, flag='val'):

        dataset = getattr(self, flag + '_data') if hasattr(self, flag + '_data') else None
        if dataset is None: return {}
        if self.fr_high_g_list != None:
            return self.calculator.test(model, dataset, self.adj_low_un, self.adj_low, self.adj_high, min(self.test_batch_size, len(dataset)), self.option['num_workers'], self.fr_high_g_list, self.fr_low_g_list, self.mlp_list)
        else:
            return self.calculator.test(model, dataset, self.adj_low_un, self.adj_low, self.adj_high, min(self.test_batch_size, len(dataset)), self.option['num_workers'])

    def unpack(self, received_pkg):

        return received_pkg['model']

    def reply(self, svr_pkg):

        model = self.unpack(svr_pkg)
        self.model_g = copy.deepcopy(model)
        self.train(model)
        cpkg = self.pack(model)
        return cpkg

    def pack(self, model, *args, **kwargs):

        return {
            "model": model,
            "model_g": model
        }

    def is_idle(self):

        return self.gv.simulator.client_states[self.id] == 'idle'

    def is_dropped(self):

        return self.gv.simulator.client_states[self.id] == 'dropped'

    def is_working(self):

        return self.gv.simulator.client_states[self.id] == 'working'

    def train_loss(self, model):

        return self.test(model, 'train')['loss']

    def val_loss(self, model):

        return self.test(model)['loss']

    def register_server(self, server=None):

        self.register_objects([server], 'server_list')
        if server is not None:
            self.server = server

    def set_local_epochs(self, epochs=None):

        if epochs is None: return
        self.epochs = epochs
        self.num_steps = self.epochs * math.ceil(len(self.train_data) / self.batch_size)
        return

    def set_batch_size(self, batch_size=None):

        if batch_size is None: return
        self.batch_size = batch_size

    def set_learning_rate(self, lr=None):

        self.learning_rate = lr if lr else self.learning_rate

    def get_time_response(self):

        return np.inf if self.dropped else self.time_response

    def get_batch_data(self):

        if self._train_loader is None:
            self._train_loader = self.calculator.get_dataloader(self.train_data, batch_size=self.batch_size,
                                                                   num_workers=self.loader_num_workers,
                                                                   pin_memory=self.option['pin_memory'], drop_last=not self.option.get('no_drop_last', False))
        try:
            batch_data = next(self.data_loader)
        except Exception as e:
            self.data_loader = iter(self._train_loader)
            batch_data = next(self.data_loader)
        # clear local_movielens_recommendation DataLoader when finishing local_movielens_recommendation training
        self.current_steps = (self.current_steps + 1) % self.num_steps
        if self.current_steps == 0:
            self.data_loader = None
            self._train_loader = None
        return batch_data

    def update_device(self, dev):

        self.device = dev
        self.calculator = self.gv.TaskCalculator(dev, self.calculator.optimizer_name)
