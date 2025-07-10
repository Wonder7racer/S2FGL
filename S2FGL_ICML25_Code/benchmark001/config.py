import torch.nn
import dgl
from load_data.load_data import load_dataset
from torch.nn import init, Parameter
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn

train_data = load_dataset('citeseer', 0.6, 0.2, 0.2, 0)

device = f"cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

def convert_to_dglgraph(data_batch):
    edge_src, edge_dst = data_batch.edge_index
    num_nodes_edges = max(edge_src.max(), edge_dst.max()).item() + 1
    if hasattr(data_batch, 'x'):
        num_nodes_features = data_batch.x.size(0)
    else:
        num_nodes_features = 0

    num_nodes = max(num_nodes_edges, num_nodes_features)

    g = dgl.graph((edge_src, edge_dst), num_nodes=num_nodes)

    if num_nodes_features > 0:
        g.ndata['feature'] = data_batch.x

    if hasattr(data_batch, 'y'):
        g.ndata['label'] = data_batch.y

    if hasattr(data_batch, 'train_mask'):
        g.ndata['train_mask'] = data_batch.train_mask
    if hasattr(data_batch, 'val_mask'):
        g.ndata['val_mask'] = data_batch.val_mask
    if hasattr(data_batch, 'test_mask'):
        g.ndata['test_mask'] = data_batch.test_mask
    return g


def assign_pseudo_labels_to_dglgraph(g, train_nid, labels):
    device = labels.device

    num_nodes = g.number_of_nodes()
    label_unk = (torch.ones(num_nodes, device=device) * -1).long()
    label_unk[train_nid] = labels.long()
    g.ndata['label_unk'] = label_unk
    return g


def create_block_from_dglgraph(g):
    block = dgl.to_block(g, dst_nodes=torch.arange(g.number_of_nodes()))
    return block


class ACMModule(nn.Module):
    def attention3(self, output_low, output_high, output_mlp):
        T = 3
        output_low, output_high, output_mlp = (
            self.layer_norm_low(output_low),
            self.layer_norm_high(output_high),
            self.layer_norm_mlp(output_mlp),
        )
        low_product = torch.mm(output_low, self.att_vec_low)
        high_product = torch.mm(output_high, self.att_vec_high)
        mlp_product = torch.mm(output_mlp, self.att_vec_mlp)
        concatenated = torch.cat([low_product, high_product, mlp_product], dim=1)
        sigmoid_result = torch.sigmoid(concatenated)
        mm_result = torch.mm(sigmoid_result, self.att_vec)
        logits = mm_result / T
        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def __init__(self,
                 input_channels: int, hidden_channels: int, output_channels: int, batch_size: int,
                 dropout=0, tail_activation=False, activation=nn.ReLU(inplace=True), gn=False):
        super(ACMModule, self).__init__()
        device = torch.device('cuda')
        self.weight_low, self.weight_high, self.weight_mlp = (
            Parameter(init.xavier_uniform_(torch.FloatTensor(hidden_channels, output_channels).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(hidden_channels, output_channels).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(hidden_channels, output_channels).to(device)))
        )
        self.layer_norm_low, self.layer_norm_high, self.layer_norm_mlp = (
            nn.LayerNorm(output_channels),
            nn.LayerNorm(output_channels),
            nn.LayerNorm(output_channels),
        )
        self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * output_channels, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * output_channels, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * output_channels, 1).to(device))),
        )
        self.att_vec = Parameter(init.xavier_uniform_(torch.FloatTensor(3, 3).to(device)))

    def forward(self, input, adj_low, adj_high, adj_low_unnormalized):
        leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        ab = torch.mm(input, self.weight_low)
        output_low = leaky_relu(torch.spmm(adj_low, ab))
        output_high = leaky_relu(torch.spmm(adj_high, torch.mm(input, self.weight_high)))
        output_mlp = leaky_relu(torch.mm(input, self.weight_mlp))

        self.att_low, self.att_high, self.att_mlp = self.attention3(
            (output_low), (output_high), (output_mlp)
        )
        return 3 * (

                self.att_low * output_low
                + self.att_high * output_high
                + self.att_mlp * output_mlp
        )

def get_S2FGL(nfeat, nhid, nclass, nlayers):
    return ACM(nfeat,
        nhid,
        nclass,
        nlayers,
        dropout=0.1,
        model_type='acmgcn',
        variant=False,
        init_layers_X=1,)

class GraphConvolution(Module):
    def __init__(
        self,
        in_features,
        out_features,
        model_type,
        output_layer=0,
        variant=False,
    ):
        super(GraphConvolution, self).__init__()
        (
            self.in_features,
            self.out_features,
            self.output_layer,
            self.model_type,
            self.variant,
        ) = (
            in_features,
            out_features,
            output_layer,
            model_type,
            variant,
        )
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        self.weight_low, self.weight_high, self.weight_mlp = (
            Parameter(init.xavier_uniform_(torch.FloatTensor(in_features, out_features).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(in_features, out_features).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(in_features, out_features).to(device))),
        )
        self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
        )
        self.layer_norm_low, self.layer_norm_high, self.layer_norm_mlp = (
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
        )
        self.layer_norm_struc_low, self.layer_norm_struc_high = nn.LayerNorm(
            out_features
        ), nn.LayerNorm(out_features)
        self.att_struc_low = Parameter(
            torch.FloatTensor(1 * out_features, 1).to(device)
        )
        self.att_vec_3 = init.xavier_uniform_(Parameter(torch.FloatTensor(3, 3).to(device)))
        self.balance_w = nn.Sigmoid()

    def reset_parameters(self):

        stdv = 1.0 / math.sqrt(self.weight_mlp.size(1))
        std_att = 1.0 / math.sqrt(self.att_vec_mlp.size(1))
        std_att_vec = 1.0 / math.sqrt(self.att_vec.size(1))

        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.struc_low.data.uniform_(-stdv, stdv)

        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)
        self.att_struc_low.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

        self.layer_norm_low.reset_parameters()
        self.layer_norm_high.reset_parameters()
        self.layer_norm_mlp.reset_parameters()
        self.layer_norm_struc_low.reset_parameters()
        self.layer_norm_struc_high.reset_parameters()

    def attention3(self, output_low, output_high, output_mlp):
        T = 3
        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_low), self.att_vec_low),
                            torch.mm((output_high), self.att_vec_high),
                            torch.mm((output_mlp), self.att_vec_mlp),
                        ],
                        1,
                    )
                ),
                self.att_vec_3,
            )
            / T
        )
        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def forward(self, input, adj_low, adj_high, adj_low_unnormalized, graph=None, fr_high_g=None, fr_low_g=None, mlp_g=None):
        output_low = F.relu(torch.spmm(adj_low, (torch.mm(input, self.weight_low))))
        output_high = F.relu(torch.spmm(adj_high, (torch.mm(input, self.weight_high))))
        output_mlp = F.relu(torch.mm(input, self.weight_mlp))
        self.att_low, self.att_high, self.att_mlp = self.attention3(
            (output_low), (output_high), (output_mlp)
        )
        return 3 * (
            self.att_low * output_low
            + self.att_high * output_high
            + self.att_mlp * output_mlp
        )

class ACM(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        nlayers,
        dropout,
        model_type,
        variant=False,
        init_layers_X=1,
    ):
        super(ACM, self).__init__()
        self.bn = nn.BatchNorm1d(nhid)
        self.preprocessed = False
        self.gcns = nn.ModuleList()
        self.model_type, self.nlayers = model_type, nlayers

        self.gcns.append(
            GraphConvolution(
                nfeat,
                nhid,
                model_type=model_type,
                variant=variant,
            )
        )
        self.gcns.append(
            GraphConvolution(
                nhid,
                nhid,
                model_type=model_type,
                variant=variant,
            )
        )
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.fea_param = init.xavier_uniform_(Parameter(torch.FloatTensor(1, 1).to(device)))
        self.xX_param = init.xavier_uniform_(Parameter(torch.FloatTensor(1, 1).to(device)))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)

    def preprocess(self, data):
        g = convert_to_dglgraph(data)
        train_nid = torch.nonzero(data.train_mask, as_tuple=False).squeeze()
        labels = data.y[train_nid]
        g = assign_pseudo_labels_to_dglgraph(g, train_nid, labels)
        self.block = create_block_from_dglgraph(g)
        self.h = data.x

    def forward(self, data, adj_low_un, adj_low, adj_high, fr_low_g=None, fr_high_g=None, mlp_g=None):
        x = data.x
        x = F.dropout(x, self.dropout, training=self.training)
        fea1 = self.gcns[0](x, adj_low, adj_high, adj_low_un)
        fea1 = F.relu(fea1)
        fea1 = F.dropout(fea1, self.dropout, training=self.training)
        node_features = self.gcns[1](fea1, adj_low, adj_high, adj_low_un)
        output = self.fc(node_features)
        return F.log_softmax(output, dim=1), node_features
