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
newcomers_size = [5, 15, 35, 70, 126, 210]
newcomers_index = [1, 5, 20, 55, 125, 251]


def lstsq_ridge(A, b, given_lambda):
    return np.linalg.solve(A.T @ A + given_lambda*np.eye(A.shape[1]), A.T @ b)

def fit(D, lambda_):
    x_folds = []
    y_folds = []
    nb_dots = data_x.shape[0]
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

# def build_X(D, X):
#     someMatrix = np.zeros((data_x.shape[0], newcomers_size[D-1]))
#     for k in range(data_x.shape[0]):
#         print(k)
#         newcomers = np.array([])
#         previous_newcomers = X[k, newcomers_index[D-2]:]
#         # print(X[k, newcomers_index[D-2]:])
#         # print(previous_newcomers.shape)
#         for i in range(nb_features_ini):
#             # for j in range(i+ (D-2)*(nb_features_ini+1), newcomers_size[D-2]):
#             for j in range(0, newcomers_size[D-2]):
#                 newcomers = np.append(newcomers, data_x[k, i] * previous_newcomers[j])
#         # print(newcomers.shape)
#         # print(len(someMatrix[k]))
#         # print(someMatrix.shape)
#         newcomers = list(set(newcomers))
#         someMatrix[k] = newcomers
#     return np.c_[X, someMatrix]

# def assemble_feature(x, D):
#     n_feature = x.shape[1]
#     Q = [(np.ones(x.shape[0]), 0, 0)]
#     # print(Q)
#     i = 0
#     # print(Q[0])
#     while Q[i][1] < D:
#         cx, degree, last_index = Q[i]
#         for j in range(last_index, n_feature):
#             Q.append((cx * x[:, j], degree + 1, j))
#         # print(Q)
#         i += 1
#     return np.column_stack([q[0] for q in Q])


def assemble_feature(x, D):
    if D == 1:
        someMatrix = np.ones(data_x.shape[0])
        return np.c_[someMatrix, x]

    elif D == 2:
        someMatrix = np.zeros((data_x.shape[0], newcomers_size[D-1]))
        for a in range(data_x.shape[0]):
            newcomers = []
            for i in range(nb_features_ini):
                for j in range(i, nb_features_ini):
                    newcomers.append(data_x[a][i]*data_x[a][j])
            someMatrix[a] = np.array(newcomers)
        return np.c_[x, someMatrix]
    elif D == 3:
        someMatrix = np.zeros((data_x.shape[0], newcomers_size[D-1]))
        for a in range(data_x.shape[0]):
            newcomers = []
            for i in range(nb_features_ini):
                for j in range(i, nb_features_ini):
                    for k in range(j, nb_features_ini):
                        newcomers.append(data_x[a][i]*data_x[a][j]*data_x[a][k])
            someMatrix[a] = np.array(newcomers)
        return np.c_[x, someMatrix]
    elif D == 4:
        someMatrix = np.zeros((data_x.shape[0], newcomers_size[D-1]))
        for a in range(data_x.shape[0]):
            newcomers = []
            for i in range(nb_features_ini):
                for j in range(i, nb_features_ini):
                    for k in range(j, nb_features_ini):
                        for l in range(k, nb_features_ini):
                            newcomers.append(data_x[a][k]*data_x[a][i]*data_x[a][j]*data_x[a][l])
            someMatrix[a] = np.array(newcomers)
        return np.c_[x, someMatrix]
    elif D == 5:
        someMatrix = np.zeros((data_x.shape[0], newcomers_size[D-1]))
        for a in range(data_x.shape[0]):
            newcomers = []
            for i in range(nb_features_ini):
                for j in range(i, nb_features_ini):
                    for k in range(j, nb_features_ini):
                        for l in range(k, nb_features_ini):
                            for m in range(l, nb_features_ini):
                                newcomers.append(data_x[a][k]*data_x[a][i]*data_x[a][j]*data_x[a][l]*data_x[a][m])
            someMatrix[a] = np.array(newcomers)
        return np.c_[x, someMatrix]
    elif D == 6:
        someMatrix = np.zeros((data_x.shape[0], newcomers_size[D-1]))
        for a in range(data_x.shape[0]):
            newcomers = []
            for i in range(nb_features_ini):
                for j in range(i, nb_features_ini):
                    for k in range(j, nb_features_ini):
                        for l in range(k, nb_features_ini):
                            for m in range(l, nb_features_ini):
                                for n in range(m, nb_features_ini):
                                    newcomers.append(data_x[a][k]*data_x[a][i]*data_x[a][j]*data_x[a][l]*data_x[a][m]*data_x[a][n])
            someMatrix[a] = np.array(newcomers)
        return np.c_[x, someMatrix]



def main():
    np.set_printoptions(precision=11)
    Etrain = np.zeros((KD, len(LAMBDA)))
    Evalid = np.zeros((KD, len(LAMBDA)))
    X = np.c_[np.ones(data_x.shape[0]), data_x]

    global feat_x
    feat_x = data_x
    for D in range(1, KD+1):
        print(D)
        feat_x = assemble_feature(feat_x, D)
        print(np.sum(feat_x[0]))

        for i in range(len(LAMBDA)):
            Etrain[D-1, i], Evalid[D-1, i] = fit(D + 1, LAMBDA[i])

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

    #we could do a last learning with these parameters on the wole set


if __name__ == "__main__":
    main()
