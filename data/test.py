# coding: utf-8

# In[1]:
import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
import argparse
import cv2
from tqdm import tqdm
# get_ipython().magic(u'matplotlib inline')
parser = argparse.ArgumentParser(description='CLTR')
parser.add_argument('--data_path',
                    type=str,
                    default='../../NWPU_CLTR',
                    help='the data path of NWPU')

args = parser.parse_args()

f = open("./NWPU_list/train.txt", "r")
train_list = f.readlines()

f = open("./NWPU_list/val.txt", "r")
val_list = f.readlines()
'''for training set'''
tmp_w, tmp_h = 0, 0
iw, ih = 0, 0
for i in tqdm(range(len(train_list))):
    fname = train_list[i].split(' ')[0] + '.jpg'
    img_path = args.data_path + '/images_2048/' + fname  # using 2048 for training

    img = cv2.imread(img_path)
    w, h = img.shape[1], img.shape[0]
    if w>tmp_w:
        tmp_w = w
        ih=h
    if h>tmp_h:
        tmp_h = h
        iw=w
print(tmp_w, ih)
print(iw,tmp_h)

'''for val set'''
tmp_w, tmp_h = 0, 0
for i in tqdm(range(len(val_list))):
    fname = val_list[i].split(' ')[0] + '.jpg'
    img_path = args.data_path + '/images/' + fname  #using 4096 for testing #4096 out of memory change to 1792

    img = cv2.imread(img_path)
    w, h = img.shape[1], img.shape[0]
    if w>tmp_w:
        tmp_w = w
        ih=h
    if h>tmp_h:
        tmp_h = h
        iw=w
print(tmp_w, ih)
print(iw,tmp_h)
