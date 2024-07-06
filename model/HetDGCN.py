import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import GCNConv, inits

from utils import ZERO, normalized_correlation_matrix
from config import MAX_TOKENS_OF_A_POST

## 这个 HetDGCN 类实现了一个异构图卷积神经网络（HetDGCN），用于处理异构图数据。该网络包含多个图卷积层，每一层都使用不同类型的图卷积操作，并动态更新邻接矩阵。

class HetDGCN(nn.Module):
    def __init__(self, args):
        super(HetDGCN, self).__init__()
        self.args = args

        self.gnn_layers = []
        self.gnn_dynamic_update_weights = []
        for _ in range(args.num_gnn_layers):
            # 初始化实体、模式和其他节点的GCN卷积层
            entity_conv = GCNConv(args.dim_node_features,
                                  args.dim_node_features, add_self_loops=False, normalize=False)
            pattern_conv = GCNConv(args.dim_node_features,
                                   args.dim_node_features, add_self_loops=False, normalize=False)
            others_conv = GCNConv(args.dim_node_features,
                                  args.dim_node_features, add_self_loops=False, normalize=False)
            self.gnn_layers.append(nn.ModuleDict(
                {'ENTITY': entity_conv, 'PATTERN': pattern_conv, 'OTHERS': others_conv}))

            # 初始化动态更新权重矩阵
            t = nn.Parameter(torch.Tensor(
                args.dim_node_features, args.dim_node_features))
            inits.glorot(t)

            self.gnn_dynamic_update_weights.append(t)

        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.gnn_dynamic_update_weights = nn.ParameterList(
            self.gnn_dynamic_update_weights)

    def forward_GCN(self, GCN, x, graphs, A, layer_num):
        if layer_num == 0:
            edge_index, edge_weight = graphs.edge_index, graphs.edge_attr
        else:
            # 更新图中的边权重
            try:
                edge_index = graphs.edge_index
                E, N = len(graphs.edge_attr), len(A)
                start = F.one_hot(edge_index[0], num_classes=N)
                end = F.one_hot(edge_index[1], num_classes=N).t()
                edge_weight = torch.diag(start.float() @ A @ end.float())
                del start, end

            except:
                print('\n[Out of Memory] There are too much edges in this batch (num = {}), so it executes as a for-loop for this batch.\n'.format(len(graphs.edge_attr)))
                edge_index = graphs.edge_index
                edges_num = len(graphs.edge_attr)
                edge_weight = torch.zeros(
                    edges_num, device=self.args.device, dtype=torch.float)

                for e in tqdm(range(edges_num)):
                    a, b = graphs.edge_index[:, e]
                    edge_weight[e] = A[a, b]

        out = GCN(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return out

    def forward(self, graphs_entity, graphs_pattern, graphs_others, nums_nodes, type2nidxs):
        # 将图数据转移到指定设备
        graphs_entity.to(self.args.device)
        graphs_pattern.to(self.args.device)
        graphs_others.to(self.args.device)

        # 获取初始特征和标准化相关矩阵
        H = torch.clone(graphs_entity.x)
        A = normalized_correlation_matrix(H)

        for i, gnn in enumerate(self.gnn_layers):
            H_entity = self.forward_GCN(
                gnn['ENTITY'], x=H, graphs=graphs_entity, A=A, layer_num=i)
            H_pattern = self.forward_GCN(
                gnn['PATTERN'], x=H, graphs=graphs_pattern, A=A, layer_num=i)
            H_others = self.forward_GCN(
                gnn['OTHERS'], x=H, graphs=graphs_others, A=A, layer_num=i)

            H = F.relu(H_entity + H_pattern + H_others)

            # 更新关联（邻接）矩阵
            A_hat = torch.sigmoid(
                H @ self.gnn_dynamic_update_weights[i] @ H.t())
            A = (1 - self.args.updated_weights_for_A) * A + self.args.updated_weights_for_A * A_hat

        map_entity = []
        map_pattern = []

        curr = 0
        for j, num in enumerate(nums_nodes):
            entity_nodes_idxs = type2nidxs[j].get('ENTITY', [])
            pattern_nodes_idxs = type2nidxs[j].get('PATTERN', [])

            curr_A = A[curr:curr+num, curr:curr+num]

            if torch.any(torch.isnan(curr_A)):
                print('curr_A: ', curr_A)

            A_sum = torch.sum(curr_A, dim=1)

            if entity_nodes_idxs:
                A_entity = curr_A[:, torch.tensor(entity_nodes_idxs)]
                map_pattern.append(A_sum - torch.sum(A_entity, dim=1))
            else:
                map_pattern.append(A_sum)

            if pattern_nodes_idxs:
                A_pattern = curr_A[:, torch.tensor(pattern_nodes_idxs)]
                map_entity.append(A_sum - torch.sum(A_pattern, dim=1))
            else:
                map_entity.append(A_sum)

            curr += num

        def _scale(t):
            if len(t) == 0:
                return t
            m, M = min(t), max(t)
            return (t - m) / (M - m + ZERO)

        map_entity = [_scale(m) for m in map_entity]
        map_pattern = [_scale(m) for m in map_pattern]

        map_entity = [m/(torch.sum(m) + ZERO) for m in map_entity]
        map_pattern = [m/(torch.sum(m) + ZERO) for m in map_pattern]

        map_entity = self.padding(map_entity)
        map_pattern = self.padding(map_pattern)

        return map_entity, map_pattern

    def padding(self, map):
        padding_map = torch.zeros(
            len(map), MAX_TOKENS_OF_A_POST, device=self.args.device, dtype=torch.float)
        for i, m in enumerate(map):
            sz = min(MAX_TOKENS_OF_A_POST, len(m))
            padding_map[i, :sz] = m[:sz]
        return padding_map
