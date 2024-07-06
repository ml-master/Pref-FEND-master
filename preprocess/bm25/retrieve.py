from gensim.summarization import bm25
import numpy as np
import json
import os
from tqdm import tqdm
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count

TOP = 10
# 计算相似度得分
def compute_bm25_scores(args):
    bm_model, query = args
    return bm_model.get_scores(query)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    dataset = args.dataset

    save_dir = 'data'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, dataset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 语料库
    with open('../../dataset/{}/raw/articles/articles.json'.format(dataset), 'r') as f:
        articles = json.load(f)
    corpus = [a['text'] for a in articles]
    print('Corpus: {}'.format(len(corpus)))

    print('\nLoading BM25 Model...')
    bm_model = bm25.BM25(corpus)
    print('Done.')

    for dataset_type in ['train', 'val', 'test']:
        with open('../../dataset/{}/raw/post/{}.json'.format(dataset, dataset_type), 'r') as f:
            pieces = json.load(f)
        queries = [p['content'] for p in pieces]
        print('\nDataset: {}, sz = {}'.format(dataset_type, len(queries)))

        bm25_scores = np.zeros((len(queries), len(corpus)))

        # Use multiprocessing to speed up scoring
        with Pool(cpu_count()) as p:
            results = list(tqdm(p.imap(compute_bm25_scores, [(bm_model, query) for query in queries]), total=len(queries)))

        for i, scores in enumerate(results):
            bm25_scores[i] = scores

        print('\nSaving the BM25 results...')
        np.save(os.path.join(save_dir, 'bm25_scores_{}_{}.npy'.format(
            dataset_type, bm25_scores.shape)), bm25_scores)
        print('Done.')

        print('\nRanking and Exporting...')
        ranked_arr = (-bm25_scores).argsort()
        arr = ranked_arr[:, :TOP]
        print('Top{}: {}'.format(TOP, arr.shape))
        np.save(os.path.join(
            save_dir, 'top{}_articles_{}.npy'.format(TOP, dataset_type)), arr)
        print('Done.')
