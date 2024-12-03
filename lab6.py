# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:45:50 2024

@author: Owl
"""

import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import data
from scipy import ndimage
import matplotlib.pyplot as plt
#from utility import segmentation_utils

def hue_eq(image):
    
    channels = [0]
    histSize = [180]
    qqq = [0, 180]
    
    hist1 = cv.calcHist([image], channels, None, histSize, qqq)
    
    lut = np.zeros([180, 1]) 
    
    hsum = hist1.sum()
    for i in range(180):
        lut[i] = np.uint8(179 * hist1[:i].sum()/hsum)
        
    image2 = image.copy()
        
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image2[i][j] = lut[image2[i][j]]
            
    return image2

def eq(image):
    
    channels = [0]
    histSize = [256]
    qqq = [0, 256]
    
    hist1 = cv.calcHist([image], channels, None, histSize, qqq)
    
    lut = np.zeros([256, 1]) 
    
    hsum = hist1.sum()
    for i in range(256):
        lut[i] = np.uint8(255 * hist1[:i].sum()/hsum)
        
    image2 = image.copy()
        
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image2[i][j] = lut[image2[i][j]]
            
    return image2




image = cv.imread('./tortik.png')
image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

h, s, v = cv.split(image_hsv)
#h = hue_eq(h)
s = eq(s)
#v = eq(v)
image_hsv = cv.merge([h, s, v])
image_to_clusterization = cv.cvtColor(image_hsv, cv.COLOR_HSV2RGB)
## Методы кластеризации. Сдвиг среднего (Mean shift)
# Сглаживаем чтобы уменьшить шум
blur_image = cv.medianBlur(image_to_clusterization,11 )
# Выстраиваем пиксели в один ряд и переводим в формат с правающей точкой
flat_image = np.float32(blur_image.reshape((-1,3)))

# Используем meanshift из библиотеки sklearn
bandwidth = estimate_bandwidth(flat_image, quantile=.075, n_samples=5000)
ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
ms.fit(flat_image)
labeled = ms.labels_

# получим количество сегментов
segments = np.unique(labeled)
print('Number of segments: ', segments.shape[0])

# получим средний цвет сегмента
total = np.zeros((segments.shape[0], 3), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
    total[label] = total[label] + flat_image[i]
    count[label] += 1
avg = total/count
avg = np.uint8(avg)
max_blue = min(avg, key = lambda i:  (i[0]/3+i[1]/3+i[2]/3))[2] #находим цвет с наибольшей долей красного
# Для каждого пискеля проставим средний цвет его сегмента
mean_shift_image = avg[labeled].reshape((image.shape))
# Маской скроем один из сегментов
mask1 = mean_shift_image[:,:,2].copy()
mask1[mask1!=max_blue] = 0 #здесь фильтрация по цвету
# в исходном изображении нас интересовал кластер, получивший цвет
# с красным каналом = 89
#TODO подобрать цвет, соотв. штанам
image_to_masking = image.copy()
image_to_masking = cv.bitwise_not(image)
mean_shift_with_mask_image = cv.bitwise_and(image_to_masking, image_to_masking, mask=mask1)
mean_shift_with_mask_image = cv.bitwise_not(mean_shift_with_mask_image)

# Построим изображение
plt.figure(figsize=(10, 10))
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(image_to_clusterization)
plt.title('Image for Clusterization')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(mean_shift_image, cmap='Set3')
plt.title('Mean Shift Image')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(cv.cvtColor(mean_shift_with_mask_image, cv.COLOR_BGR2RGB))
plt.title('Mean Shift with Mask')
plt.axis('off')
plt.show()

