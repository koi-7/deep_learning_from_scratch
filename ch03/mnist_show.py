#!/usr/bin/python3
# coding: utf-8


import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    # pil_img.show()
    pil_img.save('img.png')


def main():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False)

    img = x_train[0]
    label = t_train[0]
    print(label)

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    img_show(img)


if __name__ == '__main__':
    main()