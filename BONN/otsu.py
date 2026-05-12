import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


threshold_values = {}
h = [1]


def Hist(img):
   row, col = img.shape 
   y = np.zeros(256)
   for i in range(0,row):
      for j in range(0,col):
         y[img[i,j]] += 1
   x = np.arange(0,256)
   # plt.bar(x, y, color='b', width=5, align='center', alpha=0.25)
   # plt.show()
   return y


def regenerate_img(img, threshold):
    row, col = img.shape 
    y = np.zeros((row, col))
    for i in range(0,row):
        for j in range(0,col):
            if img[i,j] >= threshold:
                y[i,j] = 255
            else:
                y[i,j] = 0
    return y


   
def countPixel(h):
    cnt = 0
    for i in range(0, len(h)):
        if h[i]>0:
           cnt += h[i]
    return cnt


def wieght(s, e):
    w = 0
    for i in range(s, e):
        w += h[i]
    return w


def mean(s, e):
    m = 0
    w = wieght(s, e)
    for i in range(s, e):
        m += h[i] * i
    w = float(w)
    if w == 0:
        w = 0.00001
    return m/w


def variance(s, e):
    v = 0
    m = mean(s, e)
    w = wieght(s, e)
    for i in range(s, e):
        v += ((i - m) **2) * h[i]
    if w == 0:
        w = 0.00001
    v /= w
    return v
            

def threshold(h):
    cnt = countPixel(h)
    for i in range(1, len(h)):
        vb = variance(0, i)
        wb = wieght(0, i) / float(cnt)
        mb = mean(0, i)
        
        vf = variance(i, len(h))
        wf = wieght(i, len(h)) / float(cnt)
        mf = mean(i, len(h))
        
        V2w = wb * (vb) + wf * (vf)
        V2b = wb * wf * (mb - mf)**2
        
        if not math.isnan(V2w):
            threshold_values[i] = V2w


def get_optimal_threshold():
    min_V2w = min(threshold_values.values())
    optimal_threshold = [k for k, v in threshold_values.items() if v == min_V2w]
    # print ('optimal threshold', optimal_threshold[0])
    return optimal_threshold[0]


images = PSNR_15dB_dmd
noise_bin_ostu_10dB = np.zeros((6000, 280, 336))

for i in range(0, 6000):
    img = images[i]
    h = Hist(img)
    threshold(h)
    op_thres = get_optimal_threshold()
    noise_bin_ostu_10dB[i] = regenerate_img(img, op_thres)
    if (i % 100 == 0):
        plt.imshow(noise_bin_ostu_10dB[i])
        plt.show()
