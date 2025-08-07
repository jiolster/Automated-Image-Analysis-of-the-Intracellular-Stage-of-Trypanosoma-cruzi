# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 15:51:23 2025

@author: Usuario
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:11:58 2025

@author: Usuario
"""
#Librarires
import os #For importing images and saving figures
import errno # for checking whether the montage direcotry already exists

from cellpose import models, io #Cellpose ML model for single cell segmentation
from cellpose.io import imread 
from cellpose.plot import mask_overlay

import numpy as np
import matplotlib.pyplot as plt

from skimage import  measure#Extract labeled object properties
import math #Square root

import csv #Save data as csv


# Wroking direcotry (where program is saved)
wd = '/home/joaquin/Desktop/Fotos figura/AC16'
os.chdir(wd)

#Load segmented image
img=plt.imread("Dm-AC16-2-1_ch02.tif")
data = np.load("Amas_Trained_seg.npy", allow_pickle=True).item()
masks = data['masks']
outline = data['outlines']
outline_mask = outline > 0
outline[outline_mask == True] = 16000
alpha_channel = np.ones(outline.shape)
alpha_channel[outline == 0] = 0
flow = data['flows']

plt.figure(figsize = (10,10))
plt.imshow(img, cmap='gray')
plt.imshow(masks, alpha=0.5, cmap='inferno')
plt.savefig('masks_inferno.svg', format='svg')

from matplotlib.colors import ListedColormap
colors = ['red']
custom_cmap = ListedColormap(colors)
outline = np.ma.masked_where(outline_mask == False, outline_mask)
plt.figure(figsize = (10,10))
plt.imshow(img, cmap='gray')
plt.imshow(outline, cmap=custom_cmap)
plt.savefig('masks_outlines.svg', format='svg')


plt.figure(figsize = (10,10))
plt.imshow(mask_overlay(img, masks))
plt.savefig('masks_overlay.png', format='png')


