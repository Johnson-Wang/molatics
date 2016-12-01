import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sparse

def mat_dense_to_sparse(array, precision=1e-8):
    if not len(array.shape) == 2:
        array = np.reshape(array, (array.shape[-1], -1))
    non_zero = np.where(np.abs(array) > precision)
    array_sparse = coo_matrix((array[non_zero], non_zero), shape=array.shape)
    return array_sparse



def mat_dot_product(array1, array2, is_sparse=False, precision=1e-8):
    if is_sparse == False:
        return np.dot(array1, array2)
    else:
        shape = array1.shape[:-1] + array2.shape[1:]
        if not sparse.issparse(array1):
            if not len(array1.shape) == 2:
                array1 = np.reshape(array1, (-1, array1.shape[-1]))
            non_zero1 = np.where(np.abs(array1) > precision)
            array1_sparse = coo_matrix((array1[non_zero1], non_zero1), shape=array1.shape)
        else:
            array1_sparse = array1
        if not sparse.issparse(array2):
            if not len(array2.shape) == 2:
                array2 = np.reshape(array2, (array2.shape[0], -1))
            non_zero2 = np.where(np.abs(array2) > precision)
            array2_sparse = coo_matrix((array2[non_zero2], non_zero2), shape=array2.shape)
        else:
            array2_sparse = array2
        product = array1_sparse.dot(array2_sparse)
        return product.toarray().reshape(*shape)

def gaussian(a, prec=1e-6, lang="C"):
    if lang == "C":
        import _mdfc
        column = a.shape[-1]
        transform = np.zeros((column, column), dtype='double')
        independent = np.zeros(column, dtype='intc')
        num_independent = _mdfc.gaussian(transform, a.astype("double"), independent, prec)
        transform = transform[:, :num_independent]
        independent = independent[:num_independent]
        return a, transform, independent
    else:
        return gaussian_py(a, prec)

def gaussian_py(a, prec=1e-6):
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