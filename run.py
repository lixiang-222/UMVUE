# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import os
import argparse
from sklearn.metrics import roc_auc_score
np.random.seed(2020)
torch.manual_seed(2020)

import scipy.sparse as sps
import optuna
import time
import pickle
from tqdm import tqdm

from dataset import load_data
from matrix_factorization_ori import ours_icdmw_DR, UMVUE_DR, MF
from itertools import product
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU, recall_func, precision_func, f1_func

# ==================== 命令行参数解析 ====================
def parse_args():
    parser = argparse.ArgumentParser(description='UMVUE-DR Hyperparameter Optimization')
    parser.add_argument('--dataset', type=str, default='kuai',
                       choices=['coat', 'yahoo', 'kuai'],
                       help='Dataset name')
    parser.add_argument('--trials', type=int, default=1000,
                       help='Number of optimization trials')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--batch_size_prop', type=int, default=2048,
                       help='Batch size for propensity model')
    parser.add_argument('--sampling_rate', type=float, default=1.0,
                       help='Training data sampling rate (0.0 to 1.0)')
    return parser.parse_args()

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

# ==================== 工具函数 ====================

def save_results(filename, trial_number, params, results, training_time=None, epoch=None):
    """保存单个trial的结果"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'a') as file:
        file.write(f'Trial {trial_number}\n')
        
        # 保存参数（紧凑格式）
        param_str = ', '.join([f'{k}={v:.6f}' if isinstance(v, float) else f'{k}={v}' 
                               for k, v in params.items()])
        file.write(f'Params: {param_str}\n')
        
        # 保存训练信息
        if training_time is not None:
            file.write(f'Training time: {training_time:.2f}s')
            if epoch is not None:
                file.write(f', Converged epoch: {epoch}')
            file.write('\n')
        
        # 保存指标
        for key, val in results.items():
            file.write(f'{key}: {val:.6f}\n')
        
        file.write('-' * 50 + '\n\n')


def save_best_summary(filename, study):
    """保存最佳结果摘要"""
    with open(filename, 'a') as file:
        file.write('\n' + '=' * 80 + '\n')
        file.write('BEST TRIAL SUMMARY\n')
        file.write('=' * 80 + '\n')
        file.write(f'Trial number: {study.best_trial.number}\n')
        file.write(f'Best value: {study.best_trial.value:.6f}\n')
        file.write('\nBest parameters:\n')
        for key, value in study.best_trial.params.items():
            if isinstance(value, float):
                file.write(f'  {key}: {value:.6f}\n')
            else:
                file.write(f'  {key}: {value}\n')
        file.write('=' * 80 + '\n')


def generate_embeddings(x_train, y_train, x_test, y_test, num_user, num_item, 
                       embedding_k=32, dataset_name='coat', force_regenerate=False):
    """
    生成或加载用户和物品嵌入
    
    Args:
        force_regenerate: 是否强制重新生成嵌入
    
    Returns:
        u_emb, i_emb, u_emb_test, i_emb_test
    """
    emb_file = f"saved/{dataset_name}_mfdrmc.pkl"
    
    # 尝试加载已有嵌入
    if not force_regenerate:
        try:
            with open(emb_file, "rb") as f:
                u_emb = pickle.load(f)
                i_emb = pickle.load(f)
            print(f"  ✅ Loaded embeddings from {emb_file}")
            # 处理不同类型的数据
            if isinstance(u_emb, torch.Tensor):
                u_emb_test = u_emb.clone()
                i_emb_test = i_emb.clone()
            else:
                u_emb_test = u_emb.copy()
                i_emb_test = i_emb.copy()
            return u_emb, i_emb, u_emb_test, i_emb_test
        except FileNotFoundError:
            print(f"  ⚠️ Embedding file not found, generating new embeddings...")
    
    # 训练MF模型生成嵌入
    print(f"  🔨 Training MF model to generate embeddings (emb_dim={embedding_k})...")
    model = MF(num_user, num_item, batch_size=128, embedding_k=embedding_k)
    model.cuda()
    
    model.fit(
        x_train, y_train,
        lr=0.02,
        lamb=5e-4,
        tol=1e-5,
        verbose=False
    )
    
    # 评估MF模型
    test_pred = model.predict(x_test)
    mse_mf = mse_func(y_test, test_pred)
    auc_mf = roc_auc_score(y_test, test_pred)
    ndcg_res = ndcg_func(model, x_test, y_test)
    recall_res = recall_func(model, x_test, y_test)
    
    print(f"  📊 Base MF Performance:")
    print(f"     MSE: {mse_mf:.6f}, AUC: {auc_mf:.6f}")
    print(f"     NDCG@5: {np.mean(ndcg_res['ndcg_5']):.6f}, "
          f"NDCG@10: {np.mean(ndcg_res['ndcg_10']):.6f}")
    print(f"     Recall@5: {np.mean(recall_res['recall_5']):.6f}, "
          f"Recall@10: {np.mean(recall_res['recall_10']):.6f}")
    
    # 提取嵌入
    all_ui_user = np.array(list(product(np.arange(num_user), np.arange(1))))
    _, u_emb, _ = model.predict(all_ui_user, is_training=True)
    
    all_ui_item = np.array(list(product(np.arange(1), np.arange(num_item))))
    _, _, i_emb = model.predict(all_ui_item, is_training=True)
    
    # 保存嵌入
    with open(emb_file, "wb") as f:
        pickle.dump(u_emb, f)
        pickle.dump(i_emb, f)
    print(f"  💾 Saved embeddings to {emb_file}")
    
    if isinstance(u_emb, torch.Tensor):
        u_emb_test = u_emb.clone()
        i_emb_test = i_emb.clone()
    else:
        u_emb_test = u_emb.copy()
        i_emb_test = i_emb.copy()
        
    return u_emb, i_emb, u_emb_test, i_emb_test


def generate_knn_matrix(dataset_name, x_train, y_train, x_test, y_test, 
                       num_user, num_item, u_emb, i_emb, k=10, force_regenerate=False):
    """
    生成或加载KNN矩阵

    Args:
        force_regenerate: 是否强制重新生成KNN矩阵
    """

    # 尝试加载已有的KNN矩阵
    knn_file = f"saved/{dataset_name}_knn.pkl"
    if not force_regenerate:
        try:
            with open(knn_file, "rb") as f:
                knn_matrix = pickle.load(f)
            print(f"✅ Loaded existing KNN matrix from {knn_file}")
            return knn_matrix
        except FileNotFoundError:
            print(f"⚠️ KNN matrix file not found, generating new KNN matrix...")

    # 1. 准备观测矩阵
    obs = sps.csr_matrix(
        (np.ones(x_train.shape[0]), (x_train[:, 0], x_train[:, 1])),
        shape=(num_user, num_item),  # 使用训练集维度
        dtype=np.float32
    ).toarray()
    
    # 2. 生成所有user-item对
    def generate_total_sample(num_user, num_item):
        sample = []
        for i in range(num_user):
            sample.extend([[i,j] for j in range(num_item)])
        return np.array(sample)
    
    x_all = generate_total_sample(num_user, num_item)
    
    # 3. 构建嵌入矩阵
    embeddings = np.array([
        np.concatenate((u_emb[u], i_emb[i])) 
        for u, i in x_all
    ]).reshape(num_user, num_item, 64)
    
    # 4. 采样数据
    ul_idxs = np.arange(len(x_all))
    np.random.shuffle(ul_idxs)
    x_all_idx = ul_idxs[:3*len(x_train)]
    x_sampled = x_all[x_all_idx]
    x_sampled = np.r_[x_sampled, x_train]
    
    # 5. 计算KNN矩阵
    def find_k_nearest_neighbors(embeddings, k, x_train, x_sampled):
        neighbors = []
        # 确保索引是整数类型
        x_sampled_int = x_sampled.astype(int)
        x_train_int = x_train.astype(int)

        embeddings_temp = embeddings[x_sampled_int[:, 0], x_sampled_int[:, 1]]
        original_dict = {}
        for i in range(len(embeddings_temp)):
            original_dict[i] = [x_sampled_int[i][0], x_sampled_int[i][1]]
        
        # 添加进度条
        for i in tqdm(x_train_int, desc="Computing KNN neighbors", leave=False):
            distances = np.linalg.norm(embeddings_temp - embeddings[i[0], i[1]], axis=1)
            nearest_indices = np.argsort(distances)[1:k+1]
            nearest_values = [obs[original_dict[idx][0], original_dict[idx][1]] for idx in nearest_indices]
            neighbors.append(np.mean(nearest_values))
        return np.array(neighbors)
    
    knn_matrix = find_k_nearest_neighbors(embeddings, k, x_train, x_sampled)
    # knn_matrix = np.array([np.mean(obs[indices[i]]) for i in range(len(x_train))])
    
    # 6. 保存文件
    knn_file = f"{dataset_name}_ours_icdmw_knn.pkl"
    with open(knn_file, "wb") as f:
        pickle.dump(knn_matrix, f)
        f.close()

    print(f"✅ Generated and saved KNN matrix to {knn_file}")
    return knn_matrix



# ==================== 主程序 ====================
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    dataset_name = args.dataset
    n_trials = args.trials
    batch_size = args.batch_size
    batch_size_prop = args.batch_size_prop
    sampling_rate = args.sampling_rate
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    print(f"\n{'='*60}")
    print(f"🚀 UMVUE-DR Hyperparameter Optimization")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🔄 Trials: {n_trials}")
    print(f"📈 Sampling Rate: {sampling_rate}")
    print(f"{'='*60}\n")

# ==================== 数据加载 ====================

if dataset_name == "coat":
    train_mat, test_mat = load_data("coat")        
    x_train, y_train = rating_mat_to_sample(train_mat)
    x_test, y_test = rating_mat_to_sample(test_mat)
    num_user = train_mat.shape[0]
    num_item = train_mat.shape[1]

    y_train = binarize(y_train)
    y_test = binarize(y_test)

elif dataset_name == "yahoo":
    x_train, y_train, x_test, y_test = load_data("yahoo")
    x_train, y_train = shuffle(x_train, y_train)
    x_train[:, 0] -= 1
    x_train[:, 1] -= 1
    x_test[:, 0] -= 1
    x_test[:, 1] -= 1
    num_user = x_train[:,0].max() + 1
    num_item = x_train[:,1].max() + 1

    y_train = binarize(y_train)
    y_test = binarize(y_test)

elif dataset_name == "kuai":
    rdf_train = np.array(pd.read_table("./data/kuai/user.txt", header = None, sep = ','))     
    rdf_test = np.array(pd.read_table("./data/kuai/random.txt", header = None, sep = ','))
    rdf_train_new = np.c_[rdf_train, np.ones(rdf_train.shape[0])]
    rdf_test_new = np.c_[rdf_test, np.zeros(rdf_test.shape[0])]
    rdf = np.r_[rdf_train_new, rdf_test_new]
    
    rdf = rdf[np.argsort(rdf[:, 0])]
    c = rdf.copy()
    for i in range(rdf.shape[0]):
        if i == 0:
            c[:, 0][i] = i
            temp = rdf[:, 0][0]
        else:
            if c[:, 0][i] == temp:
                c[:, 0][i] = c[:, 0][i-1]
            else:
                c[:, 0][i] = c[:, 0][i-1] + 1
            temp = rdf[:, 0][i]
    
    c = c[np.argsort(c[:, 1])]
    d = c.copy()
    for i in range(rdf.shape[0]):
        if i == 0:
            d[:, 1][i] = i
            temp = c[:, 1][0]
        else:
            if d[:, 1][i] == temp:
                d[:, 1][i] = d[:, 1][i-1]
            else:
                d[:, 1][i] = d[:, 1][i-1] + 1
            temp = c[:, 1][i]

    y_train = d[:, 2][d[:, 3] == 1]
    y_test = d[:, 2][d[:, 3] == 0]
    x_train = d[:, :2][d[:, 3] == 1]
    x_test = d[:, :2][d[:, 3] == 0]
    
    num_user = x_train[:,0].max() + 1
    num_item = x_train[:,1].max() + 1

    y_train = binarize(y_train, 2)
    y_test = binarize(y_test, 2)
    num_user = int(num_user)
    num_item = int(num_item)

    print(f"[train] num data: {len(x_train)}")
    print(f"[test]  num data: {len(x_test)}")

else:
    print("Cant find the data set",dataset_name)

# ==================== 数据采样 ====================
if sampling_rate < 1.0:
    print(f"_sampling training data with rate: {sampling_rate}")
    num_samples = int(len(x_train) * sampling_rate)
    indices = np.random.choice(len(x_train), num_samples, replace=False)
    x_train = x_train[indices]
    y_train = y_train[indices]
    print(f"[sampled train] num data: {len(x_train)}")




# 需要预训练的embedding
u_emb, i_emb, u_emb_test, i_emb_test = generate_embeddings(
    x_train, y_train, x_test, y_test,
    num_user, num_item,
    embedding_k=32,
    dataset_name=dataset_name
)

knn_matrix = generate_knn_matrix(
    dataset_name, x_train, y_train, x_test, y_test,
    num_user, num_item, u_emb, i_emb, k=10
)

# 输出文件
sampling_suffix = "" if sampling_rate == 1.0 else f"_sampling{int(sampling_rate*100)}"
output_file = f'results/{dataset_name}_UMVUE_DR_results{sampling_suffix}.txt'

# ==================== Objective 函数 ====================

def objective(trial):
    # ========== 超参数采样 ==========
    params = {
        # weight decay 
        'pred_lamb': trial.suggest_float('pred_lamb', 1e-6, 5e-3),
        'impu_lamb': trial.suggest_float('impu_lamb', 1e-6, 5e-3),
        'prop_lamb': trial.suggest_float('prop_lamb', 1e-6, 5e-3),
        
        # 学习率
        'pred_lr': trial.suggest_categorical('pred_lr', [0.005, 0.01, 0.02, 0.05, 0.1]),
        'impu_lr': trial.suggest_categorical('impu_lr', [0.005, 0.01, 0.02, 0.05, 0.1]),
        'prop_lr': trial.suggest_categorical('prop_lr', [0.005, 0.01, 0.02, 0.05, 0.1]),

        # 模型结构
        'G': trial.suggest_int('G', 2, 5), # 图传播层数
        'emb': trial.suggest_categorical('emb', [4, 8, 16]),
        
        # UMVUE-DR 特有参数
        'alpha': trial.suggest_float('alpha', 0.1, 10), # 倾向性评分权重
        'beta': trial.suggest_float('beta', 0.0, 1.0), # 自相关偏差权重
        'theta': trial.suggest_float('theta', 0.0, 1.0), # 估计偏差权重
        'gamma': trial.suggest_float('gamma', 0.01, 0.1), # 噪声抑制系数
        'k': trial.suggest_categorical('k', [5, 10, 15, 20, 25]), # KNN邻居数
    }
    
    # ========== 模型训练 ==========
    model = UMVUE_DR(
        num_user, num_item,
        batch_size=batch_size,
        batch_size_prop=batch_size_prop,
        embedding_k=params['emb']
    )
    model.cuda()
    
    start = time.time()
    epoch = model.fit(
        x_train, y_train, knn_matrix, params['k'],
        G=params['G'],
        alpha=params['alpha'],
        beta=params['beta'],
        gamma=params['gamma'],
        pred_lr=params['pred_lr'],
        impu_lr=params['impu_lr'],
        prop_lr=params['prop_lr'],
        pred_lamb=params['pred_lamb'],
        impu_lamb=params['impu_lamb'],
        prop_lamb=params['prop_lamb'],
        tol=1e-5,
        verbose=False
    )
    training_time = time.time() - start
    
    # 打印训练信息
    print(f"Trial {trial.number}: {training_time:.2f}s, converged at epoch {epoch}")
    
    # ========== 模型评估 ==========
    test_pred = model.predict(x_test)
    
    # 确保是 numpy array
    if isinstance(test_pred, torch.Tensor):
        test_pred = test_pred.cpu().numpy()
    
    # ========== 根据数据集设置 top_k_list ==========
    if dataset_name in ['coat', 'yahoo']:
        top_k_list = [1, 3, 5, 10]
    else:  # kuai
        top_k_list = [10, 30, 50, 100]
    
    # 计算评估指标
    mse = mse_func(y_test, test_pred)
    auc = roc_auc_score(y_test, test_pred)
    ndcg_res = ndcg_func(model, x_test, y_test, top_k_list=top_k_list)
    recall_res = recall_func(model, x_test, y_test, top_k_list=top_k_list)
    precision_res = precision_func(model, x_test, y_test, top_k_list=top_k_list)
    f1_res = f1_func(model, x_test, y_test, top_k_list=top_k_list)
    
    # ========== 根据数据集设置目标值 ==========
    if dataset_name in ['coat', 'yahoo']:
        objective_value = auc + np.mean(ndcg_res["ndcg_5"])
        # 整理结果
        results = {
            'mse': mse,
            'auc': auc,
            'ndcg@1': np.mean(ndcg_res["ndcg_1"]),
            'ndcg@3': np.mean(ndcg_res["ndcg_3"]),
            'ndcg@5': np.mean(ndcg_res["ndcg_5"]),
            'ndcg@10': np.mean(ndcg_res["ndcg_10"]),
            'recall@1': np.mean(recall_res["recall_1"]),
            'recall@3': np.mean(recall_res["recall_3"]),
            'recall@5': np.mean(recall_res["recall_5"]),
            'recall@10': np.mean(recall_res["recall_10"]),
            'precision@1': np.mean(precision_res["precision_1"]),
            'precision@3': np.mean(precision_res["precision_3"]),
            'precision@5': np.mean(precision_res["precision_5"]),
            'precision@10': np.mean(precision_res["precision_10"]),
            'f1@1': np.mean(f1_res["f1_1"]),
            'f1@3': np.mean(f1_res["f1_3"]),
            'f1@5': np.mean(f1_res["f1_5"]),
            'f1@10': np.mean(f1_res["f1_10"]),
            'objective': objective_value
        }
    else:  # kuai
        objective_value = auc + np.mean(ndcg_res["ndcg_50"])
        # 整理结果
        results = {
            'mse': mse,
            'auc': auc,
            'ndcg@10': np.mean(ndcg_res["ndcg_10"]),
            'ndcg@30': np.mean(ndcg_res["ndcg_30"]),
            'ndcg@50': np.mean(ndcg_res["ndcg_50"]),
            'ndcg@100': np.mean(ndcg_res["ndcg_100"]),
            'recall@10': np.mean(recall_res["recall_10"]),
            'recall@30': np.mean(recall_res["recall_30"]),
            'recall@50': np.mean(recall_res["recall_50"]),
            'recall@100': np.mean(recall_res["recall_100"]),
            'precision@10': np.mean(precision_res["precision_10"]),
            'precision@30': np.mean(precision_res["precision_30"]),
            'precision@50': np.mean(precision_res["precision_50"]),
            'precision@100': np.mean(precision_res["precision_100"]),
            'f1@10': np.mean(f1_res["f1_10"]),
            'f1@30': np.mean(f1_res["f1_30"]),
            'f1@50': np.mean(f1_res["f1_50"]),
            'f1@100': np.mean(f1_res["f1_100"]),
            'objective': objective_value
        }

    # ========== 保存结果 ==========
    save_results(output_file, trial.number, params, results, training_time, epoch)
    
    return objective_value


# ==================== 运行优化 ====================

print(f"\n{'='*80}")
print(f"🚀 Starting UMVUE-DR Hyperparameter Optimization on {dataset_name}")
print(f"📈 Sampling Rate: {sampling_rate}")
print(f"{'='*80}\n")

# 禁用 Optuna 默认日志
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 创建 study
study = optuna.create_study(direction='maximize')

# 运行优化
study.optimize(objective, n_trials=n_trials)

# ==================== 输出最终结果 ====================

print('\n' + '='*80)
print('🎉 Optimization Completed!')
print('='*80)
print(f'Total trials: {len(study.trials)}')
print(f'Best value: {study.best_trial.value:.6f}')
print(f'\nBest parameters (Trial #{study.best_trial.number}):')
print('-'*80)
for key, value in study.best_trial.params.items():
    if isinstance(value, float):
        print(f'  {key:12s}: {value:.6f}')
    else:
        print(f'  {key:12s}: {value}')
print('='*80)
print(f"\n✅ Results saved to: {output_file}")

# 保存最佳结果摘要
save_best_summary(output_file, study)


# python run.py --dataset coat --batch_size 128 --trials 100  --gpu 1 --sampling_rate 0.8