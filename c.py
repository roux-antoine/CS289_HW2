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


    max_deg = 20  # max degree
    # err_train = np.zeros(n - 1)
    # err_fresh = np.zeros(n - 1)

    err_train_full = np.zeros(max_deg)
    err_fresh_full = np.zeros(max_deg)
    err_train = np.zeros(max_deg-1)
    err_fresh = np.zeros(max_deg-1)


    ## Training
    coeff = np.zeros((max_deg, len(x_train)))

    #degree 0
    coeff[0,0] = np.mean(y_train)
    err_train_full[0] = np.average((x_train - np.ones(20)*coeff[0,0])**2)

    #degree 1
    X = np.array([x_train, np.ones(len(x_train))]).flatten()
    A = X.reshape(20,2, order='F')
    coeff[1,0:2] = lstsq(A, y_train)[::-1]
    y_pred = np.zeros(len(x_train))
    for i in range(len(x_train)):
        y_pred[i] = np.dot(A[i], coeff[1][0:2])
    err_train_full[1] = np.average((y_pred - y_train)**2)

    #degree > 1
    for deg in range(2, max_deg):
        X = np.append(X[0:len(x_train)]*x_train, X)
        A = X.reshape(20,deg+1, order='F')
        coeff[deg, 0:deg+1] = lstsq(A, y_train)[::-1]

    ## computation of the trainig error
    for deg in range(max_deg):
        y_pred = A @ coeff[deg][::-1]
        err_train_full[deg] = np.average(np.array(y_train-y_pred)**2)
        err_fresh_full[deg] = np.average(np.array(y_fresh-y_pred)**2)

    err_train = err_train_full[1:]
    err_fresh = err_fresh_full[1:]

    plt.figure()
    plt.plot(range(1, max_deg), err_train, label = 'train')
    plt.plot(range(1, max_deg), err_fresh, label = 'fresh')
    plt.title('Evolution of training and validation error')
    plt.xlabel('max degree of polynomial')
    plt.ylabel('average error')
    plt.legend()
    plt.semilogy()
    plt.grid()
    plt.show()

    # plt.figure()
    # plt.ylim([0, 6])
    # plt.plot(err_train, label='train')
    # plt.plot(err_fresh, label='fresh')
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
