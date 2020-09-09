#!/usr/bin/python3
# coding: utf-8


import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


def main():
    net = simpleNet()
    print(net.W)
    print()

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print()

    print(np.argmax(p))
    print()

    t = np.array([0, 0, 1])
    print(net.loss(x, t))
    print()

    def f(W):
        return net.loss(x, t)

    dW = numerical_gradient(f, net.W)
    print(dW)



if __name__ == '__main__':
    main()

