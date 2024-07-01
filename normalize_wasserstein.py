import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,exp
import os
import pickle as pkl
from utils import fourier_threshold, wasserstein_compare, calculate_euuclidian_dist, kde
import scipy
import time
import math
import multiprocessing
import gzip


#data = np.load("/sharedata/fastdisk/nkng1/dataset_Lens.npz")

#dataset_array = data["data"]

f = gzip.GzipFile('/sharedata/fastdisk/nkng1/dataset_Lens.npz.gz', "r"); 
dataset_array = np.load(f)
f.close()
results = []

def calc_wasserstein(i):
    a = dataset_array[i[0]]
    b = dataset_array[i[1]]
    return (scipy.stats.wasserstein_distance(a[a>0], b[b > 0]))

param_array = []
for i in range(int(math.log2(len(dataset_array)))):
    for j in range(len(dataset_array)):
        param_array.append([i, j])


#print(param_array[0:100])
mypool = multiprocessing.Pool(32)



res_array = mypool.map(calc_wasserstein, param_array)
mypool.close()

print(res_array[0:10])

print(np.median(res_array))