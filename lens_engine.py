import gzip
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,exp
import os
import pickle as pkl
from utils import fourier_threshold, wasserstein_compare, calculate_euuclidian_dist, kde
import scipy
import time

#import searchengine

NORM_VALUE_256_VECTOR = 201858.73
NORM_VALUE_VECTOR = 1557815.8
NORM_VALUE_256_WASSER = 3569.5952962112524
NORM_VALUE_WASSER = 5483.749814666689

def receive_query(query_image, threshold, dataset_array,size, no, names, data_vectors, alpha=0.8):
    img = cv2.imread(query_image, 0)
    
    
    #blur method
    #img = cv2.GaussianBlur(img,(5,5),0)
    #img = cv2.blur(img,(5,5))

    img = img.astype(np.float32)
    img = cv2.resize(img, (size,size))
    print("initial sum", np.sum(img))
    img_fft = np.fft.fft2(img)
    img_thresholded = fourier_threshold(np.real(img_fft), threshold)
    img_thresholded = np.sort(np.abs((img_thresholded).flatten()))
    indices = calculate_euuclidian_dist(img_thresholded, dataset_array)
    dataset_array = dataset_array[indices]

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
    return indices_names

#grab set of lenses
lenses = []
with open("list_of_lenses.txt", "r") as f:
    lenses = f.readlines()

for i in range(0, len(lenses)):
    lenses[i] = lenses[i].strip()

#database = /sharedata/fastdisk/nkng1/QSOlens/2021-GraL-ImageSet-gri/QSO
#queries = /sharedata/fastdisk/nkng1/QSOlens/2021-GraL-ImageSet-gri/Lens



data256 = np.load("/sharedata/fastdisk/nkng1/dataset_Lens256.npz")
dataset_array256 = data256["data"]
names_array = data256["names"]
names_array = names_array[names_array != '.DS_Store']

data_vectors256 = np.load("./dataset_Lens256Vector.npz")["data"]

#------------------------------------------

f = gzip.GzipFile('/sharedata/fastdisk/nkng1/dataset_Lens.npz.gz', "r"); 
data_array_orig = np.load(f)
f.close()


#print(names_array)



n = 30
possible_queries = lenses[:n]

pos = []
number_lenses = 0

data_vectors = np.load("./dataset_LensVector.npz")["data"]

t0 = time.time()

for i in range(n):
    queryimg = '/sharedata/fastdisk/nkng1/QSOlens/2021-GraL-ImageSet-gri/QSO/'+possible_queries[i]


    indices256 = receive_query(queryimg, .95, dataset_array256, 256, 30, names_array, data_vectors256)
    #indices_names = names_array[indices256]

    data_array = data_array_orig[indices256]
    data_vectors1 = data_vectors[indices256]
    names_arrayN = names_array[indices256]


    res= np.ndarray.tolist(display_query('/home/nkng1/search engine/', queryimg, .95, data_array, names_arrayN, 1000, data_vectors1, 0.8))
    
    for l in lenses:
        if l in res:
            pos.append(res.index(l))
            number_lenses+=1/n

t1 = time.time()
print("total time: ", (t1-t0)/n)
print("average position:", np.median(np.array(pos)), " ", np.mean(np.array(pos)))
print("number of lenses per query ", number_lenses)

f = open("results.txt", "a")
f.write(res)
f.write("\n")
f.write(t1-t0)
f.close()
