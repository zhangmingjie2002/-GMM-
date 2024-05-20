'''
name:entropy.py
function:通过kmeans聚类实现基于熵增的不确定性数据选择算法
writer:ZMJ
time:2024.5.1
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_curve  
from sklearn.mixture import GaussianMixture  
import os 
import joblib  
import librosa 
import librosa.display
from sklearn.cluster import KMeans 
from extract_mfcc import extract_features
from extract_mfcc import load_audio_features
from extract_mfcc import train_gmm_with_kmeans
import shutil

#下面为初始化kmeans的随机选取的训练集子集
real_speech_dir = '/home/thesis/dataset/data_set/LA/ASVspoof2019_LA_train/select/random/random-true'  
real_features = load_audio_features(real_speech_dir,3600,'model_train_real.txt')
fake_speech_dir = '/home/thesis/dataset/data_set/LA/ASVspoof2019_LA_train/select/random/random-fake'  
fake_features = load_audio_features(fake_speech_dir,3600,'model_train_fake.txt')

#下面为训练集全集
real_test_speech_dir = '/home/thesis/dataset/data_set/LA/ASVspoof2019_LA_train/flac/bonafide'  
real_test_features = load_audio_features(real_test_speech_dir,3600,'model_train_real.txt')
fake_test_speech_dir = '/home/thesis/dataset/data_set/LA/ASVspoof2019_LA_train/flac/fake'  
fake_test_features = load_audio_features(fake_test_speech_dir,3600,'model_train_fake.txt')

#通过kmeans聚类计算熵的不确定性采样数据选择算法
def kmeans_entropy_based_sampling(features,test_features,a,n_clusters):
    '''
    features:用于初始化kmeans的随机选取的训练集子集
    test_features:训练集全集
    a:挑选的子集比例,0-1
    n_clusters:聚类个数
    '''
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features) 
    print('聚类完成')
    # 获取聚类中心  
    cluster_centers = kmeans.cluster_centers_  
  
    # 初始化一个数组来存储所有测试样本到所有聚类中心的距离  
    distances = np.zeros((test_features.shape[0], cluster_centers.shape[0]))  
  
    # 遍历每个测试样本，并计算到每个聚类中心的距离  
    for i in range(test_features.shape[0]):  
        sample = test_features[i]  
        distances[i] = np.sqrt(((sample - cluster_centers) ** 2).sum(axis=1)) 
  
    # 转换距离到“概率”  
    # 这里我们使用高斯核（或其他合适的核）将距离转换为相似度分数，  
    # 然后再通过归一化来得到“概率”  
    bandwidth = np.std(distances)  # 带宽可以是距离的标准差或其他合适的值  
    probabilities = np.exp(-(distances**2) / (2 * bandwidth**2))  
    probabilities /= probabilities.sum(axis=1, keepdims=True)  # 归一化  
  
    # 现在 probabilities 包含了每个数据点对每个聚类的“概率”分配
    entropies = []
    for i,probability in enumerate(probabilities):
        # 使用GMM预测概率 
        for j,a in enumerate(probability):
            print(f'第{j}次可能性 {a}')
        #计算熵
        entropy = -np.sum(probability * np.log2(probability + 1e-10)) 
        print(f'第{i}次entropy {entropy}')
        print('')
        #保存熵增
        entropies.append(entropy)

    num_to_select = int(0.1*len(entropies))

    test_index = np.argsort(entropies)[::-1][:num_to_select]  # 选择熵最大的num_to_select个样本
    for i,index in enumerate(test_index):
        print(f"{i} {index}")
    return test_index

#调用函数求得通过算法选择到的数据索引
real_index = kmeans_entropy_based_sampling(real_features,real_test_features,0.1,32)
fake_index = kmeans_entropy_based_sampling(fake_features,fake_test_features,0.1,32)

#存放选择出来的数据
# 创建文件夹来存放选择的样本  
folder1 = '/home/thesis/dataset/data_set/entropy_select/real'  # 存放选择出来的真实样本  
folder2 = '/home/thesis/dataset/data_set/entropy_select/fake'  # 存放选择出来的虚假样本  
os.makedirs(folder1, exist_ok=True)  
os.makedirs(folder2, exist_ok=True)

for i,index in enumerate(real_index):
    # 获取文件夹中所有文件的列表，并按顺序访问第index个文件  
    real_files = os.listdir(real_test_speech_dir)  
    source_file = os.path.join(real_test_speech_dir, real_files[index])  
          
    # 复制到目标文件夹  
    dest_folder = folder1   
    dest_file = os.path.join(dest_folder, os.path.basename(source_file))  
    shutil.copy(source_file, dest_file)

for i,index in enumerate(fake_index):
    # 获取文件夹中所有文件的列表，并按顺序访问第index个文件  
    fake_files = os.listdir(fake_test_speech_dir)  
    source_file = os.path.join(fake_test_speech_dir, fake_files[index])  
          
    # 复制到目标文件夹  
    dest_folder = folder2   
    dest_file = os.path.join(dest_folder, os.path.basename(source_file))  
    shutil.copy(source_file, dest_file)