#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_curve  
from sklearn.mixture import GaussianMixture  
import os  
import librosa 
import librosa.display
from sklearn.cluster import KMeans 
from extract_lfcc import load_audio_features
from extract_lfcc import train_gmm_with_kmeans
  
# 假设您已经有了真实语音和伪造语音的文件夹路径  
real_speech_dir = 'F:\\thesis\data_set\LA\ASVspoof2019_LA_train\select\\random\\random-true'  
fake_speech_dir = 'F:\\thesis\data_set\LA\ASVspoof2019_LA_train\select\\random\\random-fake'


# 加载真实语音和伪造语音的特征  
real_features = load_audio_features(real_speech_dir,3600)
print("Real features shape:", real_features.shape)
fake_features = load_audio_features(fake_speech_dir,3600)
print("Fake features shape:", fake_features.shape)    


# 训练高斯混合模型  
gmm_real = train_gmm_with_kmeans(real_features, 32)  
gmm_fake = train_gmm_with_kmeans(fake_features, 32) 

import joblib  

# 保存模型  
model_path_real = 'F:\\thesis\model\\real_20_32_3600_13_random_true_T.joblib'
model_path_fake = 'F:\\thesis\model\\fake_20_32_3600_13_random_fake_T.joblib'
joblib.dump(gmm_real, model_path_real)
joblib.dump(gmm_fake, model_path_fake)  