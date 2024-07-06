# Pref-FEND论文复现-机器学习大作业

1. 姓名：郭惠洪。
2. 基本说明：这个仓库为“Integrating Pattern- and Fact-based Fake News Detection via Model Preference Learning”这篇论文的复现工作。
3. 这篇论文的相关官方信息:
> **Integrating Pattern- and Fact-based Fake News Detection via Model Preference Learning.**
>
> Qiang Sheng\*, Xueyao Zhang\*, Juan Cao, and Lei Zhong.
>
> *Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM 2021)*
>
> [PDF](https://dl.acm.org/doi/10.1145/3459637.3482440) / [Poster](https://www.zhangxueyao.com/data/cikm2021-PrefFEND-poster.pdf) / [Code](https://github.com/ICTMCG/Pref-FEND)
4. bert-base-cased获取：[bert-base-cased获取](https://github.com/rohithjoginapally/bert-base-cased)。


## 数据集

论文原始实验数据集可以在“数据集”文件夹中看到，包括[Weibo Dataset](https://github.com/ICTMCG/Pref-FEND/tree/main/dataset/Weibo)和[Twitter Dataset](https://github.com/ICTMCG/Pref-FEND/tree/main/dataset/Twitter)。不过请注意，只有在提交了[“Application to Use the Datasets for Pattern- and Fact-based Joint Fake News Detection”](https://forms.office.com/r/HF00qdb3Zk)之后，您才能下载获取该数据集。

课程数据集（包括原始的和处理后的）存放在“Pref-FEND-master/dataset”中。不过，由于该数据集太大，故而没有上传到GitHub，您可以通过这个链接获取：[机器学习课程数据集获取](https://www.alipan.com/s/PqyZphvLVeq)。


## 代码相关

### Key Requirements

请参考`requirements.txt`。

### 准备工作

#### 步骤1：Stylistic Tokens & Entities Recognition

对于原始数据集而言，这一步不是必需的，因为申请得到的数据集已经在json文件中提供了已识别的结果。

对于课程数据集而言，“dataset”文件夹中的“gossip”文件夹中存放了json文件，其中提供了课程数据集的识别结果。您也可以通过`process.py`来对新的数据集进行此准备工作。

#### 步骤2：Tokenize

```
cd preprocess/tokenize
```

正如`run.sh`所示, 您需要运行:

```
python get_post_tokens.py --dataset [dataset] --pretrained_model [bert_pretrained_model]
```

#### 步骤3：Heterogeneous Graph Initialization

```
cd preprocess/graph_init
```

正如`run.sh`所示, 您需要运行:

```
python init_graph.py --dataset [dataset] --max_nodes [max_tokens_num]
```

#### 步骤4：Preparation of the Fact-based Models

注意，如果您不使用基于事实的模型作为Pref-FEND的一个组件，那么这一步就不是必需的。

##### Tokenize

```
cd preprocess/tokenize
```

正如`run.sh`所示, 您需要运行:

```
python get_articles_tokens.py --dataset [dataset] --pretrained_model [bert_pretrained_model]
```

##### Retrieve by BM25

```
cd preprocess/bm25
```

正如`run.sh`所示, 您需要运行:

```
python retrieve.py --dataset [dataset]
```

#### 步骤5：Preparation for some special fake news detectors

注意，如果您不使用“EANN-Text”或“BERT-Emo”作为Pref-FEND的一个组件，那么这一步就不是必需的。

##### EANN-Text

```
cd preprocess/EANN_Text
```

正如`run.sh`所示, 您需要运行:

```
python events_clustering.py --dataset [dataset] --events_num [clusters_num]
```

##### BERT-Emo

```
cd preprocess/BERT_Emo/code/preprocess
```

正如`run.sh`所示, 您需要运行:

```
python input_of_emotions.py --dataset [dataset]
```

### Training and Inferring

```
cd model
mkdir ckpts
```

`run_gossip.sh`中列出了训练和推理步骤中的所有配置和运行脚本。例如，如果您想在GossipCop上运行“BiLSTM(基于模式的) +DeClarE(基于事实的)”，您可以运行:

```

# BiLSTM + DeClarE (Pref-FNED)
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'gossip' \
--use_preference_map True --use_pattern_based_model True --use_fact_based_model True \
--pattern_based_model 'BiLSTM' --fact_based_model 'DeClarE' \
--lr 1e-4 --batch_size 4 --epochs 20 \
--save 'ckpts/BiLSTM+DeClarE_with_Pref-FEND'

```

之后结果将会被保存在“ckpts/BiLSTM+DeClarE_with_Pref-FEND”中。

### 实验结果

相关实验结果均保存在“Pref-FEND-main/model/ckpts”中。此外，课程大作业中也展示了相关实验结果。

## Citation

```
@inproceedings{Pref-FEND,
  author    = {Qiang Sheng and
               Xueyao Zhang and
               Juan Cao and
               Lei Zhong},
  title     = {Integrating Pattern- and Fact-based Fake News Detection via Model
               Preference Learning},
  booktitle = {{CIKM} '21: The 30th {ACM} International Conference on Information
               and Knowledge Management, Virtual Event, Queensland, Australia, November
               1 - 5, 2021},
  pages     = {1640--1650},
  year      = {2021},
  url       = {https://doi.org/10.1145/3459637.3482440},
  doi       = {10.1145/3459637.3482440}
}
```
