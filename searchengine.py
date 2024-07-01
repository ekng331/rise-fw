import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,exp
import os
import pickle as pkl
from utils import fourier_threshold, wasserstein_compare, calculate_euuclidian_dist, kde
import scipy
import time

NORM_VALUE_256_WASSER = 1088.265927
NORM_VALUE_256_VECTOR = 39118.914
NORM_VALUE_VECTOR = 302188.9
NORM_VALUE_WASSER = 1627.3175290122686


def receive_query(query_image, threshold, dataset_array,size, no, names, data_vectors, alpha=0.8):
    img = cv2.imread("./queries/"+query_image, 0)
    
    
    #blur method
    
    #img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.blur(img,(15,15))

    img = img.astype(np.float32)
    img = cv2.resize(img, (size,size))
    print("initial sum", np.sum(img))
    img_fft = np.fft.fft2(img)
    img_thresholded = fourier_threshold(np.real(img_fft), threshold)
    img_thresholded = np.sort(np.abs((img_thresholded).flatten()))
    indices = calculate_euuclidian_dist(img_thresholded, dataset_array)
    dataset_array = dataset_array[indices]

    if size == 256:
        return indices

    dataset_array = kde(dataset_array)

    #print("here1", names[indices])
    wassersteinDist, wassersteinIndices = wasserstein_compare(img_thresholded, dataset_array, no)
    wassersteinIndices = indices[wassersteinIndices]
    #print("here2", names[indices], wassersteinIndices, indices)

    #second part of calculation

    vector_horizontal = np.sum(img, axis=1)
    vector_vertical = np.sum(img, axis=0)
    final_vector = np.concatenate((vector_horizontal, vector_vertical))
    #print("sum here", np.sum(final_vector))

    euclid_dist = []
    if size == 256:
        a = NORM_VALUE_256_WASSER
        b = NORM_VALUE_256_VECTOR
    elif size == 1000:
        a = NORM_VALUE_WASSER
        b = NORM_VALUE_VECTOR
    for i in wassersteinIndices:
        vec = data_vectors[i]
        print(i, np.sum(final_vector), np.sqrt(np.sum(np.square(final_vector - vec))))
        euclid_dist.append(np.sqrt(np.sum(np.square(final_vector - vec))))
    total_distances = np.argsort(alpha * wassersteinDist/a + (1-alpha) * np.real(euclid_dist)/b)
   #print(wassersteinDist/a, 'euclid distances ', np.real(euclid_dist)/b)
    return wassersteinIndices[total_distances] 
    
    
def display_query(dataset_path, query, threshold, dataset_array, dataset_names, size, data_vectors, alpha=0.8):
    print("here display")
    indices = receive_query(query, threshold, dataset_array, size, 20, dataset_names, data_vectors, alpha)
    print("here dispaly 2")
    indices_names = dataset_names[indices]
    print(indices_names)

    '''
    plt.figure(constrained_layout=False)
    plt.xlim(0.5, 1.5)
    plt.ylim(0.5,1.5)
    plt.suptitle("Top 10 Results")
    
    for i in range(10):
        img_display =  cv2.imread(dataset_path+"/"+indices_names[i], 0)
        plt.subplot(2, 5, i+1)
        plt.title("Result #"+ str(i))
        plt.axis('off')
        plt.imshow(img_display)
    plt.show()
    
    '''
    return indices_names

queryimg = 'query1.jpg'

data = np.load("/sharedata/fastdisk/nkng1/dataset_fungiData256Sorted.npz")
dataset_array = data["data"]
names_array = data["names"]
names_array = names_array[names_array != '.DS_Store']

data_vectors = np.load("dataset_fungVectori256.npz")["data"]


t0 = time.time()
indices256 = receive_query(queryimg, .95, dataset_array, 256, 30, names_array, data_vectors)
t1 = time.time()
print("total time: ", t1-t0)
indices_names = names_array[indices256]
print("result of euclidean formula: ", indices_names.tolist().index("H1_1a_1.jpg"))



#------------------------------------------
data_array_orig = np.load("/sharedata/fastdisk/nkng1/dataset_fungiDataSorted.npz")
data_array = data_array_orig['data'][indices256]
names_array = names_array[indices256]
#print(names_array)

data_vectors = np.load("dataset_fungiVector.npz")["data"][indices256]

t0 = time.time()
n = 100
pos = []
for i in range(n):
    res= np.ndarray.tolist(display_query('/home/nkng1/search engine/', queryimg, .95, data_array, names_array, 1000, data_vectors, 0.8))
    pos.append(res.index("H1_1a_1.jpg"))

t1 = time.time()
print("total time: ", (t1-t0)/n)
print("average position:", np.median(np.array(pos)), " ", np.mean(np.array(pos)))

print(res)
