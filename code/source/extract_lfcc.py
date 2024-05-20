#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torchaudio
import librosa
from scipy.fftpack import dct
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def extract_lfcc(waveform, sample_rate, n_lfcc=20):
    # 计算STFT
    n_fft = 512
    hop_length = 160
    spec = torch.stft(waveform, n_fft, hop_length, window=torch.hamming_window(n_fft), return_complex=True)
    
    # 计算功率谱
    power_spec = spec.real**2 + spec.imag**2
    
    # 应用对数变换
    log_power_spec = torch.log1p(power_spec)
    
    # 将 torch.Tensor 转换为 numpy.array
    log_power_spec_np = log_power_spec.numpy()

    # 计算DCT以获得LFCC系数
    lfcc = dct(log_power_spec_np, type=2, norm='ortho')[:, :n_lfcc].T
    return lfcc

def extract_features(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    # 确保 waveform 是一维的
    waveform = waveform.squeeze(0)
    lfcc = extract_lfcc(waveform, sample_rate)
    
    # 使用 librosa 计算一阶和二阶差分
    delta1 = librosa.feature.delta(lfcc, order=1)
    delta2 = librosa.feature.delta(lfcc, order=2)
    
    # 合并特征
    features = np.hstack((lfcc, delta1, delta2)) 
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

def train_gmm_with_kmeans(features, n_clusters):
    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    weights = np.full(shape=n_clusters, fill_value=1.0 / n_clusters)
    covariances_type = 'full'
    
    gmm = GaussianMixture(n_components=n_clusters, weights_init=weights, means_init=kmeans.cluster_centers_,
                          covariance_type=covariances_type, random_state=0, tol=1e-13, max_iter=100, n_init=20)
    print("模型初始化结束")
    gmm.fit(features)
    
    return gmm   