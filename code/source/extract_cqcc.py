#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np  
import librosa  
import librosa.display  
from scipy.fftpack import dct
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    # CQT参数设置  
    fmin = librosa.note_to_hz('C1')  # 最低频率，例如C1  
    n_bins = 96 * 4  # 假设我们想要跨越4个八度音阶，每个八度音阶96个bins  
    bins_per_octave = 12 * 8  # 假设每个半音8个bins（为了得到更精细的分辨率）
    # 计算CQT  
    C = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)  
  
    # 对CQT取对数（模拟MFCC中的对数步骤）  
    C_log = np.log10(np.abs(C) + 1e-10)  # 添加小的数值以避免对数中的零或负值  
  
    # 对CQT的对数能量特征应用DCT变换  
    # 首先我们需要将CQT的时间帧作为特征矩阵的行  
    # 由于CQT的shape是(n_frames, n_bins)，我们不需要转置  
    cqcc = dct(C_log, type=2, axis=1, norm='ortho')[:, :20].T  # 取前20个DCT系数  
  
    # cqcc_dct现在是一个二维数组，其中每一列对应一个时间帧的20维DCT系数  
    
    delta1 = librosa.feature.delta(cqcc, order=1)
    delta2 = librosa.feature.delta(cqcc, order=2)

    # 合并特征
    features = np.hstack((cqcc, delta1, delta2)) 
    return features.flatten()

def load_audio_features(data_dir,target_length):
    all_features = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.flac'):
            path = os.path.join(data_dir, filename)
            features = extract_features(path)
            # 如果特征数组长度小于目标长度，则填充；如果大于，则截断  
            if len(features) < target_length:  
                # 使用零进行填充
                print(f'{path} 需要填充0,长度为{len(features)}')  
                all_features.append(np.pad(features, (0, target_length - len(features))))  
            else:  
                all_features.append(features[:target_length])  
            print(f'{path} 完成特征提取！')
    return np.array(all_features)  

# 基于K-means初始化GMM并训练
def train_gmm_with_kmeans(features, n_clusters):  
    # 使用K-means进行聚类  
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)  
    labels = kmeans.labels_  
    centroids = kmeans.cluster_centers_  
      
    # 根据K-means聚类结果初始化GMM参数  
    weights = np.full(shape=n_clusters, fill_value=1.0 / n_clusters)  
    covariances_type = 'full'
    '''  
    gmm = GaussianMixture(n_components=n_clusters, weights_init=weights, means_init=centroids,  
                           covariance_type=covariances_type, random_state=0)
    '''
    gmm = GaussianMixture(n_components=n_clusters, weights_init=weights, means_init=centroids,  
                       covariance_type=covariances_type, random_state=0, tol=1e-13,max_iter=100,n_init=20)  # 减小收敛阈值
    print("模型初始化结束")      
    # 使用MFCC特征训练GMM模型  
    gmm.fit(features)  
      
    return gmm   