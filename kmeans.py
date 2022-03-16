# I use the tolerance as a general guideline for comparing previous and next generation centroids. 
# If the average norm of centroids-previous_centroids is less than the tolerance, I stop.

import numpy as np
import pandas as pd

def kmeans(X:np.ndarray, k:int, centroids=None, max_iter=30, tolerance=1e-2):
    samples = X.shape[0] # 样本个数
    features = X.shape[1] # 样本特征数
    indexs = np.arange(0, samples) # 生成样本索引
    centroids_k = np.zeros((k, features))  # 初始化聚类簇中心，shape=(k, n_features)
    if centroids == 'kmeans++':
        centroids_k[0,] = X[np.random.randint(samples)] # 从数据集中随机选择一个样本点作为第一个聚类中心
            
        for centroid in range(k-1): # 从剩余样本中选择 k - 1 个聚类中心
            dists = []  # 定义一个列表存储离聚类中心最近的样本点
            
            for i in range(samples):
                point = X[i, :] # 单一样本
                min_dist = 2**32  # 初始距离
                for j in range(len(centroids_k)):
                    temp_dist = np.linalg.norm(point-centroids_k[j]) # 计算 point 与之前的每一个聚类中心的距离 
                    min_dist = min(min_dist,temp_dist)
                dists.append(min_dist)  # 存储最小距离
            max_dist = np.argmax(np.array(dists))  # 遍历完样本之后，选择距离最大的数据点作为下一个质心
            next_centroid = X[max_dist,:]
            centroids_k[centroid+1,:] = next_centroid  # 存储第二个及其之后的聚类中心
            dists = []   # dists 清零
    else:
        centroids_k = X[np.random.randint(X.shape[0], size=k)]   #随机初始化：即随机从样本中选择 k 个样本点作为初始聚类中心

    while tolerance:
        cluster_lst = [[] for i in range(k)]   # 有k个cluster
        distance_lst = [((X-centroids_k[j])**2).sum(axis=1).reshape(-1, 1) for j in range(len(centroids_k))]
        distance_arr = np.concatenate(distance_lst, axis=1)   # 横向拼接
        temp_lst = list(np.argmin(distance_arr, axis=1))
        
        for i, j in enumerate(temp_lst):
            cluster_lst[j].append(i)            
        for cluster in cluster_lst: 
            if len(cluster) == 0: 
                cluster.append(np.random.randint(X.shape[0]))
        centroid_next = np.array([sum(X[cluster])/len(X[cluster]) for cluster in cluster_lst])
        if (np.sum((centroid_next-centroids_k)**2))/len(centroid_next) < tolerance:
            tolerance = False
        centroids_k = centroid_next 
    return centroids_k, cluster_lst