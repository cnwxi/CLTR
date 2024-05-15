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
parser.add_argument('--data_path', type=str, default='../../NWPU_CLTR_TEST',
                    help='the data path of NWPU')

args = parser.parse_args()

f = open("./NWPU_list/mytrain.txt", "r")
train_list = f.readlines()

f = open("./NWPU_list/myval.txt", "r")
val_list = f.readlines()

'''for training set'''
for i in tqdm(range(len(train_list)),ncols=50):
    fname = train_list[i].split(' ')[0] + '.jpg'
    img_path = args.data_path + '/images_2048/' + fname # using 2048 for training

    img = cv2.imread(img_path)
    Img_data_pil = Image.open(img_path).convert('RGB')

    k = np.zeros((img.shape[0], img.shape[1]))
    # point_map =  np.zeros((img.shape[0], img.shape[1], 3))+255
    mat_path = img_path.replace('images', 'gt_npydata').replace('jpg', 'npy')

    with open(mat_path, 'rb') as outfile:
        gt = np.load(outfile).tolist()

    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
            # point_map = cv2.circle(point_map, (int(gt[i][0]), int(gt[i][1])), 5, (0, 0, 0), -1)

    kpoint = k.copy()

    kpoint = kpoint.astype(np.uint8)
    with h5py.File(img_path.replace('images_2048', 'gt_detr_map').replace('jpg', 'h5'), 'w') as hf:
        hf['kpoint'] = kpoint
        hf['image'] = Img_data_pil
    # print("training_part", img_path)

    # point_map = point_map + img


'''for val set'''
for i in tqdm(range(len(val_list)),ncols=50):
    fname = val_list[i].split(' ')[0] + '.jpg'
    img_path = args.data_path + '/images_2048/' + fname  # using 2048 for testing

    img = cv2.imread(img_path)
    Img_data_pil = Image.open(img_path).convert('RGB')

    k = np.zeros((img.shape[0], img.shape[1]))

    mat_path = img_path.replace('images', 'gt_npydata').replace('jpg', 'npy')

    with open(mat_path, 'rb') as outfile:
        gt = np.load(outfile).tolist()

    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
            # point_map = cv2.circle(point_map, (int(gt[i][0]), int(gt[i][1])), 5, (0, 0, 0), -1)

    kpoint = k.copy()

    kpoint = kpoint.astype(np.uint8)
    with h5py.File(img_path.replace('images_2048', 'gt_detr_map').replace('jpg', 'h5'), 'w') as hf:
        hf['kpoint'] = kpoint
        hf['image'] = Img_data_pil
