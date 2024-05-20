'''
name:gtr.py
function:实现基于托肯配比的topK数据选择算法
writer:ZMJ
time:2024.4.28
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

#下述为用于环境先验数据选择算法的预训练模型，分别利用随机的1200条数据训练出来的模型
gmm_real = joblib.load('/home/thesis/model/real_20_32_3600_13_random_true_T.joblib')
gmm_fake = joblib.load('/home/thesis/model/fake_20_32_3600_13_random_fake_T.joblib')

# 下述为训练用的真实语音和伪造语音的文件夹路径  
real_speech_dir = '/home/thesis/dataset/data_set/LA/ASVspoof2019_LA_train/flac/bonafide'  
fake_speech_dir = '/home/thesis/dataset/data_set/LA/ASVspoof2019_LA_train/flac/fake'

# 加载真实语音和伪造语音的特征  
real_features = load_audio_features(real_speech_dir,3600)  
fake_features = load_audio_features(fake_speech_dir,3600) 

#得到语句的托肯配比
def get_gtr(features,gmm):
    #存储结果的二维数组
    component_counts_per_utterance = []
    
    for i,feature in enumerate(features):
        # 使用GMM预测概率  
        probabilities = gmm.predict_proba(feature.reshape(1, -1))  
  
        # 找到每个帧概率最大的高斯分量  
        max_component_indices = np.argmax(probabilities, axis=2) 

        #计算总帧数进行归一化处理
        total_frames=max_component_indices.size

        # 计算每个高斯分量的帧计数  
        component_counts = np.bincount(max_component_indices.flatten(), minlength=gmm.n_components) / total_frames 
  
        # 保存每个语句的帧计数  
        component_counts_per_utterance.append(component_counts)
        print(f"第{i}次gtr计算完成!") 
    
    # 将列表转换为NumPy数组
    component_counts_per_utterance = np.array(component_counts_per_utterance)
    print('GTR全部计算完成!')

    return component_counts_per_utterance  

#计算二维数组中每行与其他所有行之间的平均距离,并返回行距离在前a比例的行的索引列表。 
def compute_average_distances_and_get_top_indices(component_counts_per_utterance, a):  
    """    
    参数:  
    component_counts_per_utterance (np.ndarray): 二维数组，每行代表一个语句的归一化高斯分量帧计数。  
    a (float): 0到1之间的分数,表示要返回的行比例。  
      
    返回:  
    top_indices (list): 行距离在前a比例的行的索引列表。  
    """  
    # 初始化一个数组来保存每行的平均距离  
    average_distances_per_row = []  
      
    # 遍历二维数组中的每一行  
    for i, row_i in enumerate(component_counts_per_utterance):  
        # 初始化距离总和  
        distance_sum = 0  
          
         # 遍历除了当前行以外的所有其他行  
        for j, row_j in enumerate(component_counts_per_utterance):  
            # 如果不是当前行，则计算欧几里得距离  
            if j != i:  
                distance = np.linalg.norm(row_i - row_j)  
                # 累加距离  
                distance_sum += distance  
          
        # 计算平均距离（不包括与自身的距离）  
        average_distance = distance_sum / (len(component_counts_per_utterance) - 1)   
          
        # 将平均距离添加到结果数组中  
        average_distances_per_row.append(average_distance)  
      
    # 将列表转换为NumPy数组  
    average_distances_per_row = np.array(average_distances_per_row)  
      
    # 根据平均距离对行进行排序，并获取索引  
    sorted_indices = np.argsort(average_distances_per_row)  
      
    # 计算前a比例的行数  
    num_top_rows = int(a * len(component_counts_per_utterance))  
      
    # 获取前a比例行的索引  
    top_indices = sorted_indices[-num_top_rows:]  
      
    return top_indices  

# 加载真实语音数据并获取每个语句的高斯分量帧计数并进而得到索引
real_component_counts_per_utterance = get_gtr(real_features, gmm_real)
real_index = compute_average_distances_and_get_top_indices(real_component_counts_per_utterance,0.1)

# 加载伪造语音数据并获取每个语句的高斯分量帧计数并进而得到索引  
fake_component_counts_per_utterance = get_gtr(fake_features, gmm_fake)
fake_index = compute_average_distances_and_get_top_indices(fake_component_counts_per_utterance,0.1)

#存放选择出来的数据
# 创建文件夹来存放选择的样本  
folder1 = '/home/thesis/dataset/data_set/gtr_select/real'  # 存放选择出来的真实样本  
folder2 = '/home/thesis/dataset/data_set/gtr_selectfake'  # 存放选择出来的虚假样本  
os.makedirs(folder1, exist_ok=True)  
os.makedirs(folder2, exist_ok=True)

for i,index in enumerate(real_index):
    # 获取文件夹中所有文件的列表，并按顺序访问第index个文件  
    real_files = os.listdir(real_speech_dir)  
    source_file = os.path.join(real_speech_dir, real_files[index])  
          
    # 复制到目标文件夹  
    dest_folder = folder1   
    dest_file = os.path.join(dest_folder, os.path.basename(source_file))  
    shutil.copy(source_file, dest_file)

for i,index in enumerate(fake_index):
    # 获取文件夹中所有文件的列表，并按顺序访问第index个文件  
    fake_files = os.listdir(fake_speech_dir)  
    source_file = os.path.join(fake_speech_dir, fake_files[index])  
          
    # 复制到目标文件夹  
    dest_folder = folder2   
    dest_file = os.path.join(dest_folder, os.path.basename(source_file))  
    shutil.copy(source_file, dest_file)