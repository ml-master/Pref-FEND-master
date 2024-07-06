import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from HetDGCN import HetDGCN
from PatternBasedModels import BiLSTM, BERT_Emo, EANN_Text
from FactBasedModels import DeClarE, MAC, EVIN

from config import MAX_TOKENS_OF_A_POST
from utils import ZERO


class PrefFEND(nn.Module):
    """
    这段代码主要实现了以下功能：
        初始化基于事实和基于模式的模型。
        通过 MLP 层将模型的输出进行处理。
        使用 HetDGCN 进行图卷积操作。
        实现前向传播和逆向预测。
        对节点特征进行填充，以适应模型的输入要求。
    """
    def __init__(self, args):
        super(PrefFEND, self).__init__()
        self.args = args

        # 初始化基于事实的模型（如果有指定）
        if args.use_fact_based_model:
            self.FactBasedModel = eval('{}(args)'.format(args.fact_based_model))
        else:
            args.output_dim_of_fact_based_model = 0

        # 初始化基于模式的模型（如果有指定）
        if args.use_pattern_based_model:
            self.PatternBasedModel = eval('{}(args)'.format(args.pattern_based_model))
        else:
            args.output_dim_of_pattern_based_model = 0

        # 初始化 MLP 层，用于结合模型输出
        last_output = args.output_dim_of_pattern_based_model + args.output_dim_of_fact_based_model
        self.fcs = nn.ModuleList(self.init_MLP_layers(last_output))

        # 初始化 HetDGCN（如果使用偏好图）
        if args.use_preference_map:
            self.HetDGCN = HetDGCN(args)
            self.reversed_fcs = nn.ModuleList(self.init_MLP_layers(last_output))

    def init_MLP_layers(self, last_output):
        fcs = []
        for _ in range(self.args.num_mlp_layers - 1):
            curr_output = int(last_output / 2)
            fcs.append(nn.Linear(last_output, curr_output))
            last_output = curr_output

        fcs.append(nn.Linear(last_output, self.args.category_num))
        return fcs

    def forward_PreferencedDetector(self, idxs, dataset, tokens_features, map_for_fact_detector, map_for_pattern_detector, fcs):
        fact_model_out, pattern_model_out = None, None

        if self.args.use_fact_based_model:
            fact_model_out = self.FactBasedModel(idxs, dataset, tokens_features, map_for_fact_detector)
        if self.args.use_pattern_based_model:
            pattern_model_out = self.PatternBasedModel(idxs, dataset, tokens_features, map_for_pattern_detector)

            # 处理 EANN_Text 模型的特殊情况
            if self.args.pattern_based_model == 'EANN_Text':
                pattern_model_out, event_out = pattern_model_out

        # 合并模型输出
        models_out = torch.cat([x for x in [fact_model_out, pattern_model_out] if x is not None], dim=1)

        # 通过 MLP 层
        for fc in fcs:
            models_out = F.gelu(fc(models_out))

        # 处理 EANN_Text 模型的特殊情况
        if self.args.pattern_based_model == 'EANN_Text':
            return models_out, event_out
        return models_out

    def forward(self, idxs, dataset, graphs_entity, graphs_pattern, graphs_others, nums_nodes):
        # 从图中提取节点特征
        nodes_features = []
        curr = 0
        for num in nums_nodes:
            nodes_features.append(graphs_entity.x[curr:curr+num])
            curr += num
        tokens_features = self.padding(nodes_features)

        # 获取偏好图
        if self.args.use_preference_map:
            type2nidxs = [dataset.type2nidxs[idx.item()] for idx in idxs]
            map_entity, map_pattern = self.HetDGCN(graphs_entity, graphs_pattern, graphs_others, nums_nodes, type2nidxs)
        else:
            map_entity, map_pattern = None, None

        # 正向传播
        model_out = self.forward_PreferencedDetector(idxs, dataset, tokens_features, map_for_fact_detector=map_entity, map_for_pattern_detector=map_pattern, fcs=self.fcs)

        # 处理 EANN_Text 模型的特殊情况
        if self.args.pattern_based_model == 'EANN_Text':
            model_out, event_out = model_out

        # 逆向预测
        if self.args.use_preference_map:
            model_reversed_out = self.forward_PreferencedDetector(idxs, dataset, tokens_features, map_for_fact_detector=map_pattern, map_for_pattern_detector=map_entity, fcs=self.reversed_fcs)

            # 处理 EANN_Text 模型的特殊情况
            if self.args.pattern_based_model == 'EANN_Text':
                model_reversed_out, event_reversed_out = model_reversed_out
        else:
            model_reversed_out, event_reversed_out = None, None

        # 返回结果
        if self.args.pattern_based_model == 'EANN_Text':
            return model_out, model_reversed_out, map_entity, map_pattern, event_out, event_reversed_out

        return model_out, model_reversed_out, map_entity, map_pattern

    def padding(self, nodes_features):
        # 节点特征填充
        dim = nodes_features[0].shape[-1]
        padding_nodes_features = torch.zeros((len(nodes_features), MAX_TOKENS_OF_A_POST, dim), device=self.args.device)
        mask = torch.zeros((len(nodes_features), MAX_TOKENS_OF_A_POST), device=self.args.device)
        for i, x in enumerate(nodes_features):
            sz = min(len(x), MAX_TOKENS_OF_A_POST)
            padding_nodes_features[i, :sz] = x[:sz]
            mask[i, :sz] = 1 / (sz + ZERO)

        return padding_nodes_features, mask
