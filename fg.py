import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio

data = spio.loadmat('polynomial_regression_samples.mat', squeeze_me=True)
data_x = data['x']
data_y = data['y']
Kc = 4  # 4-fold cross validation
KD = 6  # max D = 6
LAMBDA = [0, 0.05, 0.1, 0.15, 0.2]

nb_features_ini = 5
newcomers_size = [15, 35, 70, 126, 210]
newcomers_index = [1, 5, 20, 55, 125, 251]


def fit(D, lambda_):
    # YOUR CODE TO COMPUTE THE AVERAGE ERROR PER SAMPLE
    pass

def build_X(D, X):
    someMatrix = np.zeros((data_x.shape[0], newcomers_size[D-1]))
    for k in range(data_x.shape[0]):
        print(k)
        newcomers = np.array([])
        previous_newcomers = X[k, newcomers_index[D-1]:]
        for i in range(newcomers_size[D-1]):
            for j in range(i, nb_features_ini):
                newcomers = np.append(newcomers, data_x[k, i] * previous_newcomers[j])
        # print(newcomers)
        # print(len(someMatrix[k]))
        # print(someMatrix.shape)
        someMatrix[k] = newcomers
    return np.c_[X, someMatrix]




def main():
    np.set_printoptions(precision=11)
    Etrain = np.zeros((KD, len(LAMBDA)))
    Evalid = np.zeros((KD, len(LAMBDA)))
    X = np.c_[np.ones(data_x.shape[0]), data_x]

    for D in range(1, KD):
        print(D)
        X = build_X(D, X)
        print(X.shape)
        print(X[0])
        for i in range(len(LAMBDA)):
            Etrain[D, i], Evalid[D, i] = fit(D + 1, LAMBDA[i])

    print('Average train error:', Etrain, sep='\n')
    print('Average valid error:', Evalid, sep='\n')

    # YOUR CODE to find best D and i


if __name__ == "__main__":
    main()
