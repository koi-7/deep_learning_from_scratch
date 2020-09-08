#!/usr/bin/python3
# coding: utf-8


import matplotlib.pyplot as plt
from matplotlib.image import imread


img = imread('../dataset/lena.png')
plt.imshow(img)

plt.savefig('a.png')
