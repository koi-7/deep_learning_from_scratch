#!/usr/bin/python3
# coding: utf-8


import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD


def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    # 過学習を再現するために、学習データを削減
    x_train = x_train[:300]
    t_train = t_train[:300]

    network = MultiLayerNet(input_size=784, hidden_size_list = [100, 100, 100, 100, 100, 100], output_size=10)
    optimizer = SGD(lr=0.01)

    max_epochs = 201
    train_size = x_train.shape[0]
    batch_size = 100

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    for i in range(200):
        print('train_acc: ' + str(train_acc_list[i]) + ' | test_acc: ' + str(test_acc_list[i]))
        ## 訓練データに対しては完璧な精度（train_acc が 1.0）なのに対して
        ## test_acc が 0.7xxx で停滞しているのがわかる


if __name__ == '__main__':
    main()
