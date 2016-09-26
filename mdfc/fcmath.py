import numpy as np


def gaussian(a, prec=1e-6):
    """Compute the gaussian elimination with both row and column pivoting
    Parametere
    -------------------------------------------------
    a: (M, N) array_like
    array to decompose
    prec: float
    absolute values lower than prec are ignored
    Return:
    a: ((max(M, N), N) array_like
      the gaussian eliminated array
    param: (N, num_independent) array_like
      parameters which transfrom from the independent elements to all
    independent: (num_independent,) list
      indexes of independent elements"""
    dependent = []
    independent = []
    # row, column = a.shape
    row, column = a.shape
    # a = np.where(np.abs(a)<prec, 0, a).astype("double")
    a = np.array(a).astype("double")
    # if row < column:
    #     a = np.vstack((a, np.zeros((column-row, column))))
    irow = 0
    for c in range(column):
        max_column = np.max(np.abs(a), axis=0)
        indices = [m for m in range(column) if m not in dependent+independent]
        i = indices[np.argmax(max_column[indices])] # find the column of the maximum element
        for j in range(irow+1, row):
            if abs(a[j,i]) -abs(a[irow, i]) > prec:
                a[[irow, j]] = a[[j,irow]] # interchange the irowth and jth row

        if abs(a[irow, i]) > prec * 100:
            dependent.append(i)
            a[irow, indices] /= a[irow, i]
            for j in range(irow) + range(irow+1, row):
                a[j, indices] -= a[irow, indices] / a[irow, i] * a[j,i]
            irow += 1
        else:
            independent.append(i)
    param = np.zeros((column, len(independent)), dtype='double')
    if len(independent) >0:
        for i, de in enumerate(dependent):
            for j, ind in enumerate(independent):
                param[de, j] = -a[i, ind]
        for i, ind in enumerate(independent):
            param[ind, i] = 1
    return a, param, independent


def gaussian2(a, prec=1e-8):
    """Compute the gaussian elimination with row pivoting
    Parametere
    -------------------------------------------------
    a: (M, N) arraay_like
    array to decompose
    prec: float
    absolute values lower than prec are ignored
    Return:
    a: ((max(M, N), N) array_like
      the gaussian eliminated array
    param: (N, num_independent) array_like
      parameters which transfrom from the independent elements to all
    independent: (num_independent,) list
      indexes of independent elements"""
    dependent = []
    independent = []
    row, column = a.shape
    if row < column:
        a = np.vstack((a, np.zeros((column-row, column))))
    row, column = a.shape
    a = np.where(np.abs(a)<prec, 0, a).astype('double')
    irow = 0
    for i in range(column):
        for j in range(irow+1, row):
            if abs(a[j,i]) -abs(a[irow, i]) > prec:
                a[[irow, j]] = a[[j,irow]] # interchange the irowth and jth row

        if abs(a[irow, i]) > prec:
            dependent.append(i)
            a[irow, i:] /= a[irow, i]
            for j in range(irow) + range(irow+1, row):
                a[j, i:] -= a[irow, i:] / a[irow, i] * a[j,i]
            irow += 1
        else:
            independent.append(i)
    param = np.zeros((column, len(independent)), dtype='double')
    if len(independent) >0:
        for i, de in enumerate(dependent):
            for j, ind in enumerate(independent):
                param[de, j] = -a[i, ind]
        for i, ind in enumerate(independent):
            param[ind, i] = 1
    return a, param, independent


def similarity_transformation(rot, mat):
    """ R x M x R^-1 """
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))