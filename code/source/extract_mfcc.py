#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os  
import librosa  
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture  

# 特征提取函数，包含MFCC及其动态差分
def extract_features(audio_file):  
    y, sr = librosa.load(audio_file, sr=None)  
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T  
    delta1 = librosa.feature.delta(mfccs, order=1)    
    delta2 = librosa.feature.delta(mfccs, order=2)         
    features = np.hstack((mfccs, delta1, delta2))  
    return features.flatten()   
  
# 加载音频文件并提取特征，同时确保特征长度统一  
def load_audio_features(data_dir, target_length):  
    features = []
    for filename in os.listdir(data_dir):  
        if filename.endswith('.flac'):
            filepath = os.path.join(data_dir, filename)  
            # 提取特征  
            extracted_features = extract_features(audio_file=filepath)
            print(f'{filepath} 完成特征提取！');  
            # 如果特征数组长度小于目标长度，则填充；如果大于，则截断  
            if len(extracted_features) < target_length:  
                # 使用零进行填充
                print(f'{filepath} 需要填充0,长度为{len(extracted_features)}')  
                features.append(np.pad(extracted_features, (0, target_length - len(extracted_features))))  
            else:  
                features.append(extracted_features[:target_length])  
    return np.array(features)   

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