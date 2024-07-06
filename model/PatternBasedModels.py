import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.autograd import Function
from transformers import BertModel


## 这段代码实现了三种不同的模型：

## BiLSTM：用于处理序列数据的双向 LSTM 模型。
## BERT_Emo：结合 BERT 和情感特征的文本分类模型。
## EANN_Text：用于假新闻检测和事件分类的模型，包含文本卷积神经网络（TextCNN）和梯度反转层（GRL）。

class BiLSTM(nn.Module):
    def __init__(self, args):
        super(BiLSTM, self).__init__()

        self.args = args

        self.max_sequence_length = args.bilstm_input_max_sequence_length
        self.num_layers = args.bilstm_num_layer
        self.hidden_size = args.bilstm_hidden_dim

        self.lstm = nn.LSTM(args.bilstm_input_dim, self.hidden_size, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.bilstm_dropout)
        self.fc = nn.Linear(self.hidden_size * 2,
                            args.output_dim_of_pattern_based_model)

    def forward(self, idxs, dataset, tokens_features, maps=None):
        tokens_features, masks = tokens_features

        # 使用双向LSTM处理输入特征
        bilstm_out, _ = self.lstm(tokens_features)

        if maps is None:
            # 如果没有提供注意力图，使用掩码计算加权和
            attention_out = torch.sum(masks[:, :, None] * bilstm_out, dim=1)
        else:
            # 使用提供的注意力图计算加权和
            attention_out = torch.sum(maps[:, :, None] * bilstm_out, dim=1)

        # 全连接层进行分类
        out = self.fc(attention_out)
        return out



class BERT_Emo(nn.Module):
    def __init__(self, args) -> None:
        super(BERT_Emo, self).__init__()

        self.args = args

        self.bert = BertModel.from_pretrained(
            args.bert_pretrained_model, return_dict=False)

        # 设置哪些BERT层需要训练，哪些不需要
        for name, param in self.bert.named_parameters():
            if name.startswith("pooler"):
                if 'bias' in name:
                    param.data.zero_()
                elif 'weight' in name:
                    param.data.normal_(
                        mean=0.0, std=self.bert.config.initializer_range)
                param.requires_grad = True
            elif name.startswith('encoder.layer.11'):
                param.requires_grad = True
            elif name.startswith('embeddings'):
                param.requires_grad = args.bert_training_embedding_layers
            else:
                param.requires_grad = args.bert_training_inter_layers

        fixed_layers = [name for name, param in self.bert.named_parameters() if not param.requires_grad]

        print('\n', '*'*15, '\n')
        print("BERT_Emo Fixed layers: {} / {}: \n{}".format(
            len(fixed_layers), len(self.bert.state_dict()), fixed_layers))
        print('\n', '*'*15, '\n')

        self.maxlen = min(int(args.bert_input_max_sequence_length * 1.5), 512)
        self.doc_maxlen = self.maxlen - 2
        print('args.bert_hidden_dim=',args.bert_hidden_dim)
        print('args.bert_emotion_dim=',args.bert_emotion_dim)
        print('args.output_dim_of_pattern_based_model=',args.output_dim_of_pattern_based_model)


        self.fc = nn.Linear(args.bert_hidden_dim + args.bert_emotion_dim,
                            args.output_dim_of_pattern_based_model)

    def forward(self, idxs, dataset, tokens_features, maps=None):
        nodes_tokens = [dataset.graphs[idx.item()]['graph'] for idx in idxs]
        nodes_tokens = [[n[-1] for n in nodes[:self.args.bert_input_max_sequence_length]]
                        for nodes in nodes_tokens]

        if maps is not None:
            tokened_maps = torch.zeros(
                len(maps), self.maxlen, device=self.args.device, dtype=torch.float)

            for i, nodes in enumerate(nodes_tokens):
                m = maps[i]
                tokened_m = [[m[nidx]/len(node) for _ in node]
                             for nidx, node in enumerate(nodes)]
                tokened_m = [a for b in tokened_m for a in b]

                sz = min(self.maxlen, len(tokened_m))
                tokened_maps[i, :sz] = torch.as_tensor(
                    tokened_m[:sz], device=self.args.device, dtype=torch.float)
        else:
            tokened_maps = None

        nodes_tokens = [[a for b in nodes for a in b]
                        for nodes in nodes_tokens]

        out = self.forward_BERT(idxs, dataset, nodes_tokens, maps=tokened_maps)
        return out

    def forward_BERT(self, idxs, dataset, tokens, maps=None):
        inputs = [self._encode(t) for t in tokens]
        input_ids = torch.tensor(
            [i[0] for i in inputs], dtype=torch.long, device=self.args.device)
        masks = torch.stack([i[1] for i in inputs])

        seq_output, _ = self.bert(input_ids)

        if maps is None:
            semantic_output = torch.sum(masks*seq_output, dim=1)
        else:
            semantic_output = torch.sum(maps[:, :, None] * seq_output, dim=1)

        if self.args.bert_emotion_dim > 0:
            print(f"idxs: {idxs}, dataset size: {len(dataset.BERT_Emo_features)}")
            # if isinstance(idxs, torch.Tensor):
            #     idxs = idxs.tolist()
            # valid_idxs = [idx for idx in idxs if idx < len(dataset.BERT_Emo_features)]
            # print(f"valid_idxs: {valid_idxs}")
            emotion_output = dataset.BERT_Emo_features[idxs]
            # emotion_output = dataset.BERT_Emo_features[valid_idxs]
            emotion_output = torch.tensor(
                emotion_output, dtype=torch.float, device=self.args.device)
            output = torch.cat([semantic_output, emotion_output], dim=1)
        else:
            output = semantic_output
        out = self.fc(output)
        return out

    def _encode(self, doc):
        doc = doc[:self.doc_maxlen]
        padding_length = self.maxlen - (len(doc) + 2)
        input_ids = [101] + doc + [102] + [103] * padding_length
        mask = torch.zeros(self.maxlen, 1, dtype=torch.float,
                           device=self.args.device)
        mask[:-padding_length] = 1 / (len(doc) + 2)
        return input_ids, mask



class EANN_Text(nn.Module):
    def __init__(self, args):
        super(EANN_Text, self).__init__()
        self.args = args

        self.input_dim = args.eann_input_dim
        self.hidden_dim = args.eann_hidden_dim
        self.event_num = args.eann_event_num
        self.max_sequence_length = args.eann_input_max_sequence_length

        self.use_textcnn = args.eann_use_textcnn

        channel_in = 1
        filter_num = 20
        window_sizes = [1, 2, 3, 4]
        self.convs = nn.ModuleList(
            [nn.Conv2d(channel_in, filter_num, (K, self.input_dim)) for K in window_sizes])
        self.fc_cnn = nn.Linear(
            filter_num * len(window_sizes), self.hidden_dim)

        self.event_discriminator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_dim, self.event_num),
            nn.Softmax(dim=1)
        )

        self.fake_news_detector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_dim, args.output_dim_of_pattern_based_model),
        )

    def forward(self, idxs, dataset, tokens_features, maps=None):
        tokens_features, masks = tokens_features

        if maps is None:
            text = tokens_features * masks[:, :, None]
        else:
            text = tokens_features * maps[:, :, None]

        text = text.unsqueeze(1)
        text = [F.relu(conv(text)).squeeze(3) for conv in self.convs]
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)
        text = F.relu(self.fc_cnn(text))

        detector_output = self.fake_news_detector(text)

        reverse_text_feature = grad_reverse(text)
        discriminator_output = self.event_discriminator(reverse_text_feature)

        return detector_output, discriminator_output


class GRL(Function):
    @staticmethod
    def forward(ctx, x, lamd):
        ctx.lamd = lamd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.lamd
        return grad_output, None


def grad_reverse(x, lamd=1):
    return GRL.apply(x, lamd)

