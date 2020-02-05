import numpy as np
import time
import itertools


def linear_surplus_confidence_matrix(B, alpha):
    # To construct the surplus confidence matrix, we need to operate only on the nonzero elements.
    # This is not possible: S = alpha * B
    S = B.copy()
    S.data = alpha * S.data
    return S


def log_surplus_confidence_matrix(B, alpha, epsilon):
    # To construct the surplus confidence matrix, we need to operate only on the nonzero elements.
    # This is not possible: S = alpha * np.log(1 + B / epsilon)
    S = B.copy()
    S.data = alpha * np.log(1 + S.data / epsilon)
    return S


def iter_rows(S):
    """
    Helper function to iterate quickly over the data and indices of the
    rows of the S matrix. A naive implementation using indexing
    on S is much, much slower.
    """
    for i in range(S.shape[0]):
        lo, hi = S.indptr[i], S.indptr[i + 1]  # indptr - index in spasity matrix
        yield i, S.data[lo:hi], S.indices[lo:hi]  # col score index


def recompute_factors(Y, S, lambda_reg, dtype='float32'):
    """
    recompute matrix X from Y.
    X = recompute_factors(Y, S, lambda_reg)
    This can also be used for the reverse operation as follows:
    Y = recompute_factors(X, ST, lambda_reg)
    The comments are in terms of X being the users and Y being the items.
    """
    m = S.shape[0]  # m = number of users
    f = Y.shape[1]  # f = number of factors
    YTY = np.dot(Y.T, Y)  # precompute this
    YTYpI = YTY + lambda_reg * np.eye(f)
    X_new = np.zeros((m, f), dtype=dtype)

    for k, s_u, i_u in iter_rows(S):  # col score index
        Y_u = Y[i_u]  # exploit sparsity
        A = np.dot(s_u + 1, Y_u)
        YTSY = np.dot(Y_u.T, (Y_u * s_u.reshape(-1, 1)))
        B = YTSY + YTYpI

        # Binv = np.linalg.inv(B)
        # X_new[k] = np.dot(A, Binv) 
        X_new[k] = np.linalg.solve(B.T, A.T).T  # doesn't seem to make much of a difference in terms of speed, but w/e

    return X_new


def recompute_factors_V(Y, V, hV, S, sim_v, sim_hv, lambda_reg, beta, dtype='float32'):
    """
    recompute item factor matrix V
    """
    m = S.shape[0]  # m = number of POIs
    f = Y.shape[1]  # f = number of factors
    YTY = np.dot(Y.T, Y)  # precompute this
    YTYpI = YTY + lambda_reg * np.eye(f)
    X_new = np.zeros((m, f), dtype=dtype)

    # k, u
    for k, s_u, i_u in iter_rows(S):  # col score index
        lo, hi = sim_hv.indptr[k], sim_hv.indptr[k+1]
        lo1, hi1 = sim_v.indptr[k], sim_v.indptr[k + 1]
        sim_hv_k = sim_hv.data[lo:hi]  # similarity of poi-com
        sim_v_k = sim_v.data[lo1:hi1]  # similarity of poi-com
        Y_u = Y[i_u]  # exploit sparsity
        YTSY = np.dot(Y_u.T, (Y_u * s_u.reshape(-1, 1)))
        B = YTSY + YTYpI + beta * (np.sum(sim_hv_k) * np.eye(f) + np.sum(sim_v_k) * np.eye(f))
        sim_hv_matrix = sim_hv_k
        sim_v_matrix = sim_v_k
        for i in range(f-1):
            sim_hv_matrix = np.row_stack((sim_hv_matrix, sim_hv_k))
        for j in range(f-1):
            sim_v_matrix = np.row_stack((sim_v_matrix, sim_v_k))
        temp1 = np.sum(np.multiply(sim_hv_matrix.T, hV[sim_hv.indices[lo:hi]]), axis=0)
        temp2 = np.sum(np.multiply(sim_v_matrix.T, V[sim_v.indices[lo1:hi1]]), axis=0)
        A = np.dot(s_u + 1, Y_u) + temp1 + temp2

        # Binv = np.linalg.inv(B)
        # X_new[k] = np.dot(A, Binv)
        X_new[k] = np.linalg.solve(B.T, A.T).T  # doesn't seem to make much of a difference in terms of speed, but w/e

    return X_new


def recompute_factors_hV(Y, sim_hv, lambda_reg, beta, dtype='float32'):
    """
    recompute item factor matrix V
    """
    m = sim_hv.shape[1]  # m = number of commuties
    f = Y.shape[1]  # f = number of factors
    X_new = np.zeros((m, f), dtype=dtype)

    for k, s_u, i_u in iter_rows(sim_hv.T):  # col score index
        sim_hv_matrix = s_u
        for i in range(f-1):
            sim_hv_matrix = np.row_stack((sim_hv_matrix, s_u))
        up = np.sum(np.multiply(sim_hv_matrix.T, Y[i_u]), axis=0)  # multiply
        down = lambda_reg + beta * np.sum(sim_hv[i_u, k])
        X_new[k] = up / down

    return X_new


def factorize(S, sim_v, sim_hv_csr, sim_hv_csc, num_factors, lambda_reg=1e-5, beta=1e-3, num_iterations=20, init_std=0.01, verbose=False
              , dtype='float32',
              recompute_factors=recompute_factors):
    """
    factorize a given sparse matrix using the Weighted Matrix Factorization algorithm by
    Hu, Koren and Volinsky.

    S: 'surplus' confidence matrix, i.e. C - I where C is the matrix with confidence weights.
        S is sparse while C is not (and the sparsity pattern of S is the same as that of
        the preference matrix, so it doesn't need to be specified separately).

    num_factors: the number of factors.

    lambda_reg: the value of the regularization constant.

    num_iterations: the number of iterations to run the algorithm for. Each iteration consists
        of two steps, one to recompute U given V, and one to recompute V given U.

    init_std: the standard deviation of the Gaussian with which V is initialized.

    verbose: print a bunch of stuff during training, including timing information.

    dtype: the dtype of the resulting factor matrices. Using single precision is recommended,
        it speeds things up a bit.

    recompute_factors: helper function that implements the inner loop.

    returns:
        U, V: factor matrices. If bias=True, the last columns of the matrices contain the biases.
    """
    global start_time
    sim_v, sim_hv_csr, sim_hv_csc = sim_v, sim_hv_csr, sim_hv_csc
    num_users, num_items = S.shape

    if verbose:
        print("precompute transpose")
        start_time = time.time()

    ST = S.T.tocsr()

    if verbose:
        print("took %.3f seconds" % (time.time() - start_time))
        print("run ALS algorithm")
        start_time = time.time()

    np.random.seed(num_users)
    U = None  # no need to initialize U, it will be overwritten anyway
    V = np.random.randn(num_items, num_factors).astype(dtype) * init_std
    hV = np.random.randn(num_items, num_factors).astype(dtype) * init_std

    for i in range(num_iterations):
        if verbose:
            print("  iteration %d" % i)
            print("    recompute user factors U")

        U = recompute_factors(V, S, lambda_reg, dtype)

        if verbose:
            print("    time since start: %.3f seconds" % (time.time() - start_time))
            print("    recompute item factors V")

        V = recompute_factors_V(U, V, hV, ST, sim_v, sim_hv_csr, lambda_reg, beta, dtype)

        if verbose:
            print("    time since start: %.3f seconds" % (time.time() - start_time))
            print("    recompute community factors hV")

        hV = recompute_factors_hV(V, sim_hv_csc, lambda_reg, beta, dtype)

        if verbose:
            print("    time since start: %.3f seconds" % (time.time() - start_time))

    return U, V
