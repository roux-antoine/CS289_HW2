#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio


#There is numpy.linalg.lstsq, whicn you should use outside of this classs
def lstsq(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)


def main():
    data = spio.loadmat('1D_poly.mat', squeeze_me=True)
    x_train = np.array(data['x_train'])
    y_train = np.array(data['y_train']).T
    y_fresh = np.array(data['y_fresh']).T


    n = 20  # max degree
    # err_train = np.zeros(n - 1)
    # err_fresh = np.zeros(n - 1)

    err_train = np.zeros(n)
    err_fresh = np.zeros(n)

    ## Training
    coeff = np.zeros((len(x_train), n))

    #degree 0
    coeff[0,0] = np.mean(y_train)
    err_train[0] = np.sum((x_train - np.ones(20)*coeff[0,0])*(x_train - np.ones(20)*coeff[0,0]))

    #degree 1
    X = np.array([x_train, np.ones(len(x_train))]).flatten()
    A = X.reshape(20,2, order='F')
    coeff[1,0:2] = lstsq(A, y_train)[::-1]

    y_pred = np.zeros(len(x_train))
    for i in range(len(x_train)):
        y_pred[i] = np.dot(A[i], coeff[1][0:2])

    err_train[1] = np.sum((y_pred - y_train)*(y_pred - y_train))

    #degree > 1
    for k in range(2, n):
        X = np.append(X[0:len(x_train)]*x_train, X)
        A = X.reshape(20,k+1, order='F')
        coeff[k, 0:k+1] = lstsq(A, y_train)[::-1]

    #la matrice A a l'air bien construite

    ## computation of the training error
    for k in range(n):
        y_pred = np.zeros(len(x_train))
        for i in range(len(x_train)):
            y_pred[i] = np.dot(A[i], coeff[k][::-1])
        # plt.plot(x_train, y_pred)
        # plt.plot(x_train, y_train)
        # plt.show()
        err_train[k] = np.sum(np.array(y_train-y_pred)*np.array(y_train-y_pred))


    ## computation of the 'fresh' error

    for k in range(n):
        y_pred = np.zeros(len(x_train))
        for i in range(len(x_train)):
            y_pred[i] = np.dot(A[i], coeff[k][::-1])
        # plt.plot(x_train, y_pred)
        # plt.plot(x_train, y_fresh)
        # plt.title(k)
        # plt.show()
        err_fresh[k] = np.sum(np.array(y_fresh-y_pred)*np.array(y_fresh-y_pred))

    plt.figure()
    plt.plot(err_train, label = 'train')
    plt.plot(err_fresh, label = 'fresh')
    plt.title('Evolution of training and validation error')
    plt.xlabel('max degree of polynomial')
    plt.ylabel('average error')
    plt.semilogy()
    plt.show()

    # plt.figure()
    # plt.ylim([0, 6])
    # plt.plot(err_train, label='train')
    # plt.plot(err_fresh, label='fresh')
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
