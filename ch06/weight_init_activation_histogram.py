#!/usr/bin/python3
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    x = np.random.randn(1000, 100)
    node_num = 100
    hidden_layer_size = 5
    activatoins = {}

    for i in range(hidden_layer_size):
        if i != 0:
            x = activatoins[i-1]

        # w = np.random.randn(node_num, node_num) * 1
        # w = np.random.randn(node_num, node_num) * 0.01
        w = np.random.randn(node_num, node_num) / np.sqrt(node_num)

        z = np.dot(x, w)
        a = sigmoid(z)
        activatoins[i] = a

    # ヒストグラムを描画
    for i, a in activatoins.items():
        plt.subplot(1, len(activatoins), i+1)
        plt.title(str(i+1) + '-layer')
        plt.hist(a.flatten(), 30, range=(0, 1))
    plt.savefig('activations.png')


if __name__ == '__main__':
    main()
