'''
name:env_first.py
function:实现环境先验的数据选择算法
writer:ZMJ
time:2024.4.26
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

# 合并特征和标签  
all_features = np.concatenate((real_features, fake_features))  
labels = np.concatenate((np.zeros(len(real_features)), np.ones(len(fake_features))))  
  
# 计算对数似然值  
log_likelihood_real = gmm_real.score_samples(all_features)  
log_likelihood_fake = gmm_fake.score_samples(all_features)  
  
# 计算决策分数（对数似然差）  
decision_scores = log_likelihood_fake - log_likelihood_real  
  
# 计算 ROC 曲线  
fpr, tpr, thresholds = roc_curve(labels, decision_scores)  
  
# 计算假负率 FNR  
fnr = 1 - tpr  
  
# 查找 FPR 和 FNR 曲线交点的索引  
intersection_idx = np.argmin(np.abs(fpr - fnr))  
  
# 提取交点的纵轴值（等错误率）  
equal_error_rate = fpr[intersection_idx]  
  
# 找到对应的阈值  
err_threshold = thresholds[intersection_idx]  
  
# 创建文件夹来存放误判的样本  
folder1 = '/home/thesis/dataset/data_set/select/fake'  # 存放被误判为真实的虚假语音  
folder2 = '/home/thesis/dataset/data_set/select/real'  # 存放被误判为虚假的真实语音  
os.makedirs(folder1, exist_ok=True)  
os.makedirs(folder2, exist_ok=True)  
  
# 根据阈值来区分哪些样本被误判，并将对应的文件名存储到对应的列表中  
misclassified_fake_as_real = []  
misclassified_real_as_fake = []  
  
# 假设 decision_scores 和 labels 是按顺序提取的特征值和对应的标签  
for i, (score, label) in enumerate(zip(decision_scores, labels)):  
    if label == 0 and score >= err_threshold:  # 真实语音被误判为虚假语音  
        # 获取文件夹中所有文件的列表，并按顺序访问第i个文件  
        real_files = os.listdir(real_speech_dir)  
        source_file = os.path.join(real_speech_dir, real_files[i])  
          
        # 复制到目标文件夹  
        dest_folder = '/home/thesis/dataset/data_set/select/real'  # 误判的真实语音文件的目标文件夹  
        dest_file = os.path.join(dest_folder, os.path.basename(source_file))  
        shutil.copy(source_file, dest_file)  
          
        # 记录误判的文件路径（如果需要）  
        misclassified_real_as_fake.append(source_file)  
    elif label == 1 and score < err_threshold:  # 虚假语音被误判为真实语音  
        # 获取文件夹中所有文件的列表，并按顺序访问第i个文件  
        fake_files = os.listdir(fake_speech_dir)  
        source_file = os.path.join(fake_speech_dir, fake_files[i-2580])  
          
        # 复制到目标文件夹  
        dest_folder = '/home/thesis/dataset/data_set/select/fake'  # 误判的虚假语音文件的目标文件夹  
        dest_file = os.path.join(dest_folder, os.path.basename(source_file))  
        shutil.copy(source_file, dest_file)  
          
        # 记录误判的文件路径（如果需要）  
        misclassified_fake_as_real.append(source_file)

print(f"ERR 为: {equal_error_rate}")  
print(f"被误判为真实的虚假语音已存放到 {folder1}")  
print(f"被误判为虚假的真实语音已存放到 {folder2}")