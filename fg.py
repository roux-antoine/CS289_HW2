import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
# import time

data = spio.loadmat('polynomial_regression_samples.mat', squeeze_me=True)
data_x = data['x']
data_y = data['y']
Kc = 4  # 4-fold cross validation
KD = 6  # max D = 6
LAMBDA = [0, 0.05, 0.1, 0.15, 0.2]

nb_features_ini = 5
nb_points = data_x.shape[0]
newcomers_size = [5, 15, 35, 70, 126, 210]

def lstsq_ridge(A, b, given_lambda):
    return np.linalg.solve(A.T @ A + given_lambda*np.eye(A.shape[1]), A.T @ b)

def fit(D, lambda_):
    x_folds = []
    y_folds = []
    nb_dots = nb_points
    for k in range(Kc):
        x_folds.append(feat_x[int(k*nb_dots/4) : int((k+1)*nb_dots/4)])
        y_folds.append(data_y[int(k*nb_dots/4) : int((k+1)*nb_dots/4)])

    E_train = np.zeros(Kc)
    E_valid = np.zeros(Kc)
    for k in range(Kc):
        #learning
        X = np.r_[x_folds[k%Kc], x_folds[(k+1)%Kc], x_folds[(k+2)%Kc]]
        y = np.r_[y_folds[k%Kc], y_folds[(k+1)%Kc], y_folds[(k+2)%Kc]]
        alpha = lstsq_ridge(X, y, lambda_)

        #validation
        E_train[k] = np.mean((y - X @ alpha)*(y - X @ alpha))
        E_valid[k] = np.mean((y_folds[(k+3)%Kc] - x_folds[(k+3)%Kc] @ alpha)*(y_folds[(k+3)%Kc] - x_folds[(k+3)%Kc] @ alpha))

    return ( (1/Kc)* np.sum(E_train), (1/Kc)* np.sum(E_valid) )


def create_x(x, D):
    if D == 1:
        someMatrix = np.ones(nb_points)
        return np.c_[someMatrix, x]

    elif D == 2:
        someMatrix = np.zeros((nb_points, newcomers_size[D-1]))
        for a in range(nb_points):
            newcomers = []
            for i in range(nb_features_ini):
                data_i = data_x[a][i]
                for j in range(i, nb_features_ini):
                    newcomers.append(data_i*data_x[a][j])
            someMatrix[a] = np.array(newcomers)
        return np.c_[x, someMatrix]
    elif D == 3:
        someMatrix = np.zeros((nb_points, newcomers_size[D-1]))
        for a in range(nb_points):
            newcomers = []
            for i in range(nb_features_ini):
                data_i = data_x[a][i]
                for j in range(i, nb_features_ini):
                    data_j = data_x[a][j]
                    for k in range(j, nb_features_ini):
                        newcomers.append(data_i*data_j*data_x[a][k])
            someMatrix[a] = np.array(newcomers)
        return np.c_[x, someMatrix]
    elif D == 4:
        someMatrix = np.zeros((nb_points, newcomers_size[D-1]))
        for a in range(nb_points):
            newcomers = []
            for i in range(nb_features_ini):
                data_i = data_x[a][i]
                for j in range(i, nb_features_ini):
                    data_j = data_x[a][j]
                    for k in range(j, nb_features_ini):
                        data_k = data_x[a][k]
                        for l in range(k, nb_features_ini):
                            newcomers.append(data_i*data_j*data_k*data_x[a][l])
            someMatrix[a] = np.array(newcomers)
        return np.c_[x, someMatrix]
    elif D == 5:
        someMatrix = np.zeros((nb_points, newcomers_size[D-1]))
        for a in range(nb_points):
            newcomers = []
            for i in range(nb_features_ini):
                data_i = data_x[a][i]
                for j in range(i, nb_features_ini):
                    data_j = data_x[a][j]
                    for k in range(j, nb_features_ini):
                        data_k = data_x[a][k]
                        for l in range(k, nb_features_ini):
                            data_l = data_x[a][l]
                            for m in range(l, nb_features_ini):
                                newcomers.append(data_i*data_j*data_k*data_l*data_x[a][m])
            someMatrix[a] = np.array(newcomers)
        return np.c_[x, someMatrix]
    elif D == 6:
        someMatrix = np.zeros((nb_points, newcomers_size[D-1]))
        for a in range(nb_points):
            newcomers = []
            for i in range(nb_features_ini):
                data_i = data_x[a][i]
                for j in range(i, nb_features_ini):
                    data_j = data_x[a][j]
                    for k in range(j, nb_features_ini):
                        data_k = data_x[a][k]
                        for l in range(k, nb_features_ini):
                            data_l = data_x[a][l]
                            for m in range(l, nb_features_ini):
                                data_m = data_x[a][m]
                                for n in range(m, nb_features_ini):
                                    newcomers.append(data_i*data_j*data_k*data_l*data_m*data_x[a][n])
            someMatrix[a] = np.array(newcomers)
        return np.c_[x, someMatrix]


def main():
    # start = time.time()
    np.set_printoptions(precision=11)
    Etrain = np.zeros((KD, len(LAMBDA)))
    Evalid = np.zeros((KD, len(LAMBDA)))
    X = np.c_[np.ones(nb_points), data_x]

    global feat_x
    feat_x = data_x
    for D in range(1, KD+1):
        print(D)
        feat_x = create_x(feat_x, D)
        print(np.sum(feat_x[0]))

        for i in range(len(LAMBDA)):
            Etrain[D-1, i], Evalid[D-1, i] = fit(D, LAMBDA[i])

    print('Average train error:', Etrain, sep='\n')
    print('Average valid error:', Evalid, sep='\n')

    #finding minima
    min = np.inf
    argmin_D = np.inf
    for k in range(Evalid.shape[0]):
        if np.min(Evalid[k]) < min:
            min = np.min(Evalid[k])
            argmin_D = k
    print('Dopt = ', argmin_D+1)

    min = np.inf
    argmin_lambda = np.inf
    for k in range(Evalid.shape[1]):
        if np.min(Evalid[:,k]) < min:
            min = np.min(Evalid[:,k])
            argmin_lambda = k
    print('lambda opt = ', LAMBDA[argmin_lambda])

    # print('TIME: ', time.time()-start)

    #we could do a last learning with these parameters on the whole set


if __name__ == "__main__":
    main()
