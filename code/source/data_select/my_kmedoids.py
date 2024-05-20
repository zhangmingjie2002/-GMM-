'''
name:my_kmedoids.py
function:实现kmedoids聚类数据选择算法
writer:ZMJ
time:2024.5.2
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os  
from sklearn_extra.cluster import KMedoids  
import numpy as np  
import librosa  
import shutil
from extract_mfcc import extract_features
from extract_mfcc import load_audio_features 

# 真伪语音训练集的原始语音文件夹和输出文件夹路径  
original_real_audio_folder = '/home/thesis/dataset/data_set/LA/ASVspoof2019_LA_train/flac/bonafide'  
output_real_audio_folder = '/home/thesis/dataset/data_set/kmedoid/real'
original_fake_audio_folder = '/home/thesis/dataset/data_set/LA/ASVspoof2019_LA_train/flac/fake'  
output_fake_audio_folder = '/home/thesis/dataset/data_set/kmedoid/fake' 
  
# 如果输出文件夹已存在，则删除它（可选）  
if os.path.exists(output_real_audio_folder):  
    shutil.rmtree(output_real_audio_folder)  
os.makedirs(output_real_audio_folder)  

if os.path.exists(output_fake_audio_folder):  
    shutil.rmtree(output_fake_audio_folder)  
os.makedirs(output_fake_audio_folder) 

# 加载音频特征  
real_features = load_audio_features(original_real_audio_folder, 3600)  
fake_features = load_audio_features(original_fake_audio_folder, 3600)  

# 应用K-Medoids聚类  
real_kmedoids = KMedoids(n_clusters=256, random_state=0).fit(real_features)  
real_labels = real_kmedoids.labels_  # 获取每个样本的簇标签
fake_kmedoids = KMedoids(n_clusters=2560, random_state=0).fit(fake_features)  
fake_labels = fake_kmedoids.labels_  # 获取每个样本的簇标签  
  
# 初始化一个字典来存储每个簇的medoid索引  
real_medoid_indices = {}
fake_medoid_indices = {}  
  
# 找到每个簇的medoid（这里假设medoid是簇中心最近的点）  
for i in range(real_kmedoids.n_clusters):  
    # 获取簇i的所有样本的索引  
    cluster_indices = np.where(real_labels == i)[0]  
    # 计算簇i中每个样本到簇中心的距离（这里假设使用欧氏距离）  
    distances = np.linalg.norm(real_features[cluster_indices] - real_kmedoids.cluster_centers_[i], axis=1)  
    # 找到距离最小的样本的索引，即medoid的索引  
    medoid_index = cluster_indices[np.argmin(distances)]  
    # 将medoid的索引存储在字典中  
    real_medoid_indices[i] = medoid_index

for i in range(fake_kmedoids.n_clusters):  
    # 获取簇i的所有样本的索引  
    cluster_indices = np.where(fake_labels == i)[0]  
    # 计算簇i中每个样本到簇中心的距离（这里假设使用欧氏距离）  
    distances = np.linalg.norm(fake_features[cluster_indices] - fake_kmedoids.cluster_centers_[i], axis=1)  
    # 找到距离最小的样本的索引，即medoid的索引  
    medoid_index = cluster_indices[np.argmin(distances)]  
    # 将medoid的索引存储在字典中  
    fake_medoid_indices[i] = medoid_index  
  
# 复制选定的音频文件到输出文件夹  
real_filenames = os.listdir(original_real_audio_folder)  
for cluster_id, medoid_index in real_medoid_indices.items():  
    # 这里假设real_filenames是按照real_features中的样本顺序排序的  
    filename = real_filenames[medoid_index]  
    shutil.copy(os.path.join(original_real_audio_folder, filename), output_real_audio_folder)

print(f'已将{len(real_medoid_indices)}个音频文件复制到{output_real_audio_folder}')

fake_filenames = os.listdir(original_fake_audio_folder)  
for cluster_id, medoid_index in fake_medoid_indices.items():  
    # 这里假设fake_filenames是按照fake_features中的样本顺序排序的  
    filename = fake_filenames[medoid_index]
    print(original_fake_audio_folder+filename) 
    shutil.copy(os.path.join(original_fake_audio_folder, filename), output_fake_audio_folder)

print(f'已将{len(fake_medoid_indices)}个音频文件复制到{output_fake_audio_folder}')