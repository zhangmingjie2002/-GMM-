#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_curve  
from sklearn.mixture import GaussianMixture  
import os  
import librosa 
import librosa.display 
import joblib
from scipy.interpolate import interp1d 
from extract_mfcc import extract_features
from extract_mfcc import load_audio_features

gmm_real = joblib.load('F:\\thesis\model\\real_20_32_3600_13_random_true_T.joblib')
gmm_fake = joblib.load('F:\\thesis\model\\fake_20_32_3600_13_random_fake_T.joblib')

# 真实语音和伪造语音的测试数据路径  
real_test_speech_dir = 'F:\\thesis\data_set\LA\ASVspoof2019_LA_dev\\flac\\bonafide'
fake_test_speech_dir = 'F:\\thesis\data_set\LA\ASVspoof2019_LA_dev\\flac\\fake'  

# 加载真实语音和伪造语音的特征  
real_features = load_audio_features(real_test_speech_dir,3600)  
fake_features = load_audio_features(fake_test_speech_dir,3600) 

# 合并特征和标签  
all_features = np.concatenate((real_features, fake_features))  
labels = np.concatenate((np.zeros(len(real_features)), np.ones(len(fake_features))))
 
# 计算对数似然值  
log_likelihood_real = gmm_real.score_samples(all_features)  
log_likelihood_fake = gmm_fake.score_samples(all_features)  

#打印得分
# 将分数列表转换为NumPy数组，以便于后续操作  
real_scores_array = np.array(log_likelihood_real)
fake_scores_array = np.array(log_likelihood_fake)  

# 打开一个文件以写入模式  
with open('scores.txt', 'w') as f:  
    # 将每个分数写入文件，每个分数占一行  
    for a1, a2 in zip(real_scores_array, fake_scores_array):  
        # 使用字符串连接  
        f.write(str(a1) + " " + str(a2) + "\n")

# 计算决策分数（对数似然差）  
decision_scores = log_likelihood_fake - log_likelihood_real 
  
# 计算ROC曲线  
fpr, tpr, thresholds = roc_curve(labels, decision_scores)  
  
# 计算假负率FNR  
fnr = 1 - tpr  

# 查找 FPR 和 FNR 曲线交点的索引  
intersection_idx = np.argmin(np.abs(fpr - fnr))  
  
# 提取交点的纵轴值（等错误率）  
equal_error_rate = fpr[intersection_idx]  # 或者 fnr[intersection_idx]，因为它们是相等的  
  
# 打印交点的纵轴值  
print(f"ERR为:{equal_error_rate}")  
  
# 绘制 FPR 和 FNR 曲线  
plt.figure(figsize=(10, 5))  
plt.plot(thresholds, fpr, label='FPR', color='blue')  
plt.plot(thresholds, fnr, label='FNR', color='red')  
  
# 在交点上添加标记  
plt.scatter(thresholds[intersection_idx], equal_error_rate, color='black', zorder=3, label='Equal Error Rate Point')  
  
# 设置图表属性  
plt.xlabel('Threshold')  
plt.ylabel('Rate')  
plt.title('FPR and FNR Curves')  
plt.legend()  
plt.grid(True)  
plt.tight_layout()  
  
# 显示图表  
plt.show()
