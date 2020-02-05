# -*- coding:utf-8 -*-

import wmf_search
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from time import time
import scipy.io as sio
import sys
import hyperopt
from hyperopt import fmin, space_eval, rand, anneal


COUNT = 0


def joint_model(train, test, ratings, matrix_geo):
    # calculate the pre/recall
    preN = [0]
    recallN = [0]
    userN = len(ratings)
    rating = max_min_normalization(ratings) # normalization
    mat_geo = max_min_normalization(matrix_geo)
    for w in range(11):
        w = 0.1 * w
        reconstruct_matrix = (1 - w) * rating + w * mat_geo

        # calculate the pre/recall
        pre5 = 0
        recall5 = 0
        reconstruct_matrix[train > 0] = 0
        for i in range(userN):
            Vu = np.mat(test[i]).nonzero()[1]
            rec = np.argsort(-reconstruct_matrix[i])
            hit5 = np.sum([i in Vu for i in rec[0:5]])
            pre5 += hit5 / 5
            recall5 += hit5 / len(Vu)
        preN.append(pre5)
        recallN.append(recall5)
    pre5 = max(preN) / userN
    recall5 = max(recallN) / userN

    print("Max: pre5:", pre5, "recall5:", recall5)

    return pre5, recall5


def max_min_normalization(data_value):
    """ Data normalization using max value and min value

    Args:
        data_value: The data to be normalized
        data_col_max_values: The maximum value of data's columns
        data_col_min_values: The minimum value of data's columns
    """
    data_col_max_values = data_value.max(axis=1)
    data_shape = data_value.shape
    data_rows = data_shape[0]
    # data_cols = data_shape[1]

    for i in range(data_rows):
        # for j in xrange(0, data_cols, 1):
        data_value[i] = data_value[i] / data_col_max_values[i]

    return data_value


def eva_train(args):
    global COUNT
    COUNT += 1
    num_factors, beta, lambda_reg = args
    lambda_reg = 0.0001 * (10**lambda_reg)
    print("Training iter:" , COUNT)
    print("P:num_factors=" + str(num_factors) + ", lambda_reg=" + str(lambda_reg) + ", beta=" + str(beta)
          + ", num_iterations=" + str(num_iterations) + ", init_std=0.01:\n")
    U, V = wmf_search.factorize(S, sim_v, sim_hv_csr, sim_hv_csc, num_factors=int(num_factors), lambda_reg=lambda_reg,
                                beta=beta, num_iterations=10, init_std=0.01, verbose=True, dtype='float32')
    ratings = np.dot(U, V.T)
    pre5, recall5 = joint_model(RATE_MATRIX_train, RATE_MATRIX_test, ratings, matrix_geo)

    return 1-pre5


if __name__ == "__main__":
    # error 0x00000005 => change to python3
    data = sio.loadmat(r'F:\dataming\Code\SocialMF_visual\Geo\Yelp_byGeo_P5_2_G2.mat')
    # data = sio.loadmat(r'F:\dataming\Code\SocialMF_visual\Geo\BT_byGeo_P8_2.mat')

    # data_geo = sio.loadmat('F:\dataming\Code\SocialMF_visual\Geo\Geo_clean\BT_predict.mat')
    data_geo = sio.loadmat('F:\dataming\Code\SocialMF_visual\Geo\Geo_clean\Yelp_predict_2.mat')
    matrix_geo = data_geo['predict_matrix']

    RATE_MATRIX_train = data['train_y_mat']
    RATE_MATRIX_test = data['test_y_mat']
    C = csr_matrix(RATE_MATRIX_train)
    sim = sio.loadmat(r'similarity0_Yelp.mat')
    # sim = sio.loadmat(r'similarity0_BT.mat')
    sim_v = sim['sim_v']
    sim_v = csr_matrix(sim_v)
    sim_hv = sim['sim_hv']

    # Once similarity set to 1
    # sim_v[sim_v > 0] = 1
    # sim_hv[sim_hv > 0] = 1
    ####

    sim_hv_csr = csr_matrix(sim_hv)
    sim_hv_csc = csc_matrix(sim_hv)
    lambda_reg = 1e-1
    num_factors = 40
    beta = 80
    num_iterations = 10
    global n_count
    n_count = 0

    S = wmf_search.log_surplus_confidence_matrix(C, alpha=2.0, epsilon=1e-6)
    # f = open("model_parameter_hyperopt.txt", 'a')

    hp = hyperopt.hp
    space = [hp.quniform('num_factors', 10, 100, 10), hp.quniform('beta', 10, 100, 10), hp.quniform('lambda_reg', 0, 4, 1)]
    best = fmin(eva_train, space, algo=anneal.suggest, max_evals=200)
    print(best)
    print('Finished!')
    print('no set to 1')
