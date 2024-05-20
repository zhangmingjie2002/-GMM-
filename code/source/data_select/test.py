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

speech_dir = '/home/thesis/dataset/data_set/LA/ASVspoof2019_LA_train/select/random/random-fake'  
features = load_audio_features(speech_dir,3600,'model_train_real.txt')

test_speech_dir = '/home/thesis/dataset/data_set/LA/test/'  
test_features = load_audio_features(test_speech_dir,3600,'model_train_real.txt')
kmeans = KMeans(n_clusters=6, random_state=0).fit(features) 
print('聚类完成')
# 获取聚类中心  
cluster_centers = kmeans.cluster_centers_  
  
# 计算每个数据点到所有聚类中心的距离  
distances = np.sqrt(((test_features - cluster_centers) ** 2).sum(axis=1))  
  
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

num_to_select = int(a*len(entropies))

test_index = np.argsort(entropies)[::-1][:num_to_select]  # 选择熵最大的num_to_select个样本
for i,index in enumerate(test_index):
    print(f"{i} {index}")

'''
#基于熵增的不确定性数据采样算法,返回比例a大小的最不确定的样本索引
def entropy_based_sampling(features,kmeans,a):
    entropies = []
    for i,feature in enumerate(features):
        # 使用GMM预测概率 
        probabilities = kmeans.predict_proba(feature.reshape(1, -1))
        for j,probability in enumerate(probabilities):
          print(f'第{j}次可能性 {probability}')
        #计算熵
        entropy = -np.sum(probabilities[0] * np.log2(probabilities[0] + 1e-10)) 
        print(f'第{i}次entropy {entropy}')
        print('')
        #保存熵增
        entropies.append(entropy)

    num_to_select = int(a*len(entropies))

    selected_indices = np.argsort(entropies)[::-1][:num_to_select]  # 选择熵最大的num_to_select个样本
    return selected_indices 

test_index = entropy_based_sampling(test_features,kmeans,1)
for i,index in enumerate(test_index):
    print(f"{i} {index}")
'''