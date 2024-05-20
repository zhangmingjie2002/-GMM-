'''
name:coreset.py
function:实现核心集挑选数据选择算法
writer:ZMJ
time:2024.5.6
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import glob
import joblib
import shutil
from extract_mfcc import load_audio_features
from extract_mfcc import train_gmm_with_kmeans

#计算 X 中每个点的灵敏度
def compute_sensitivity(X, B, alpha):
    distances = np.linalg.norm(X[:, np.newaxis] - B, axis=2) ** 2
    min_distances = np.min(distances, axis=1)
    cluster_assignment = np.argmin(distances, axis=1)
    sensitivities = np.zeros(X.shape[0])

    for j in range(B.shape[0]):
        cluster_points = X[cluster_assignment == j]
        if cluster_points.size > 0:
            cluster_distances = np.linalg.norm(cluster_points - B[j], axis=1) ** 2
            avg_cluster_distance = np.mean(cluster_distances)
            for i in np.where(cluster_assignment == j)[0]:
                sensitivities[i] = alpha * min_distances[i] + 2 * alpha * (avg_cluster_distance + np.mean(min_distances))
    return sensitivities

#根据灵敏度从 X 中抽样核心集
def sample_coreset(X, sensitivities, m):
    probabilities = sensitivities / np.sum(sensitivities)
    indices = np.random.choice(X.shape[0], size=m, p=probabilities)
    weights = 1 / (m * probabilities[indices])
    return X[indices], weights, indices

# 核心集构建和抽样,X是特征值，k是聚类个数，alpha是近似因子，coreset_size是核心集大小
def construct_coreset(X, k, alpha, coreset_size):
    kmeans = KMeans(n_clusters=k, init='k-means++').fit(X)
    B = kmeans.cluster_centers_
    sensitivities = compute_sensitivity(X, B, alpha)
    coreset, weights, indices = sample_coreset(X, sensitivities, coreset_size)
    return coreset, weights, indices

#基于加权EM初始化GMM模型(待修改)
def weighted_gmm_fit(data, weights, n_components, max_iter=100):
    # 初始化GMM
    gmm = GaussianMixture(n_components=n_components, max_iter=max_iter, random_state=0)
    # 使用数据和对应权重拟合模型
    gmm.fit(data, weights=weights)
    return gmm

# 真伪语音训练集的原始语音文件夹和输出文件夹路径  
original_real_audio_folder = '/home/thesis/dataset/data_set/LA/ASVspoof2019_LA_train/flac/bonafide'  
output_real_audio_folder = '/home/thesis/dataset/data_set/coreset/real'
original_fake_audio_folder = '/home/thesis/dataset/data_set/LA/ASVspoof2019_LA_train/flac/fake'  
output_fake_audio_folder = '/home/thesis/dataset/data_set/coreset/fake' 

# 如果输出文件夹已存在，则删除它（可选）  
if os.path.exists(output_real_audio_folder):  
    shutil.rmtree(output_real_audio_folder)  
os.makedirs(output_real_audio_folder)  

if os.path.exists(output_fake_audio_folder):  
    shutil.rmtree(output_fake_audio_folder)  
os.makedirs(output_fake_audio_folder) 

#提取特征值和得到核心集
real_features= load_audio_features(original_real_audio_folder, 3600)
real_coreset, real_weights, real_indices = construct_coreset(real_features, k=32, alpha=16, coreset_size=258)
gmm_real = train_gmm_with_kmeans(real_coreset, n_clusters=32)

#保存模型
model_path_real = '/home/thesis/model/real_coreset'
joblib.dump(gmm_real, model_path_real)

print("真实模型训练完成")

#保存挑选出来的真实语音,复制选定的音频文件到输出文件夹  
real_filenames = os.listdir(original_real_audio_folder)  
for index in real_indices:  
    # 这里假设real_filenames是按照real_features中的样本顺序排序的  
    filename = real_filenames[index]  
    shutil.copy(os.path.join(original_real_audio_folder, filename), output_real_audio_folder)

print(f'已将{len(real_indices)}个音频文件复制到{output_real_audio_folder}')

#提取特征值和得到核心集
fake_features= load_audio_features(original_fake_audio_folder, 3600)
fake_coreset, fake_weights, fake_indices = construct_coreset(fake_features, k=32, alpha=16, coreset_size=2280)
gmm_fake = gmm_real = train_gmm_with_kmeans(fake_coreset, n_clusters=32)

#保存模型
model_path_fake = '/home/thesis/model/fake_coreset'
joblib.dump(gmm_fake, model_path_fake)

print("虚假模型训练完成")

#保存挑选出来的虚假语音,复制选定的音频文件到输出文件夹  
fake_filenames = os.listdir(original_fake_audio_folder)  
for index in fake_indices:    
    filename = fake_filenames[index]  
    shutil.copy(os.path.join(original_fake_audio_folder, filename), output_fake_audio_folder)

print(f'已将{len(fake_indices)}个音频文件复制到{output_fake_audio_folder}')