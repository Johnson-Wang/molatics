#include <Python.h>
#include <numpy/arrayobject.h>
#include "lapack_wrapper.h"
#include "fc2.h"
#include "fc3.h"
#define warning_print(...) fprintf(stderr, __VA_ARGS__)

static PyObject * py_phonopy_pinv(PyObject *self, PyObject *args);
static PyObject * py_phonopy_pinv_mt(PyObject *self, PyObject *args);
static PyObject * py_get_fc3_spg_invariance(PyObject *self, PyObject *args);
static PyObject * py_get_fc3_coefficients(PyObject *self, PyObject *args);
static PyObject * py_get_fc3_coefficients_triplet(PyObject *self, PyObject *args);
static PyObject * py_rearrange_disp_fc3(PyObject *self, PyObject *args);
static PyObject * py_rearrange_disp_fc2(PyObject *self, PyObject *args);
static PyObject * py_test(PyObject *self, PyObject *args);
static PyObject * py_gaussian(PyObject *self, PyObject *args);
static PyMethodDef functions[] = {
  {"pinv", py_phonopy_pinv, METH_VARARGS, "Pseudo-inverse using Lapack dgesvd"},
  {"pinv_mt", py_phonopy_pinv_mt, METH_VARARGS, "Multi-threading pseudo-inverse using Lapack dgesvd"},
  {"test",py_test, METH_VARARGS, "testing"},
  {"gaussian", py_gaussian, METH_VARARGS, "Gaussian elimination"},
  {"get_fc3_coefficients", py_get_fc3_coefficients, METH_VARARGS, "Obtain the transformation matrix from irreducible fc3 (<=27) triplets to the whole"},
  {"get_fc3_coefficients_triplet", py_get_fc3_coefficients_triplet, METH_VARARGS, "Obtain the transformation matrix from irreducible fc3 (<=27) triplets to a given triplet"},
  {"get_fc3_spg_invariance", py_get_fc3_spg_invariance, METH_VARARGS, "Obtain the transformation matrix from irreducible fc3 (<=27) components to full (27) for each irreducible fc3 unit"},
  {"rearrange_disp_fc3", py_rearrange_disp_fc3, METH_VARARGS, "Rearrange the displacements as the coefficients of fc3"},
  {"rearrange_disp_fc2", py_rearrange_disp_fc2, METH_VARARGS, "Rearrange the displacements as the coefficients of fc2"},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_mdfc(void)
{
  Py_InitModule3("_mdfc", functions, "C-extension for mdfc\n\n...\n");
  return;
}


static PyObject * py_phonopy_pinv(PyObject *self, PyObject *args)
{
  PyArrayObject* data_in_py;
  PyArrayObject* data_out_py;
  double cutoff;

  if (!PyArg_ParseTuple(args, "OOd",
      &data_in_py,
      &data_out_py,
      &cutoff)) {
    return NULL;
  }

  const int m = (int)data_in_py->dimensions[0];
  const int n = (int)data_in_py->dimensions[1];
  const double *data_in = (double*)data_in_py->data;
  double *data_out = (double*)data_out_py->data;
  int info;
  
  info = phonopy_pinv(data_out, data_in, m, n, cutoff);

  return PyInt_FromLong((long) info);
}

static PyObject * py_phonopy_pinv_mt(PyObject *self, PyObject *args)
{
  PyArrayObject* data_in_py;
  PyArrayObject* data_out_py;
  PyArrayObject* row_nums_py;
  PyArrayObject* info_py;
  int max_row_num, column_num;
  double cutoff;

  if (!PyArg_ParseTuple(args, "OOOiidO",
      &data_in_py,
      &data_out_py,
      &row_nums_py,
      &max_row_num,
      &column_num,
      &cutoff,
      &info_py)) {
    return NULL;
  }

  const int *row_nums = (int*)row_nums_py->data;
  const int num_thread = (int)row_nums_py->dimensions[0];
  const double *data_in = (double*)data_in_py->data;
  double *data_out = (double*)data_out_py->data;
  int *info = (int*)info_py->data;
  
  phonopy_pinv_mt(data_out,
      info,
      data_in,
      num_thread,
      row_nums,
      max_row_num,
      column_num,
      cutoff);

  Py_RETURN_NONE;
}

static PyObject * py_get_fc3_spg_invariance(PyObject *self, PyObject *args)
{
  Py_Initialize();
  import_array();
  PyObject *tuple_return;
  PyArrayObject* triplets_reduced_py;
  PyArrayObject* positions_py;
  PyArrayObject* rotations_atom1_py;
  PyArrayObject* translations_atom1_py;
  PyArrayObject* mappings_atom1_py;
  PyArrayObject* rotations_atom2_py;
  PyArrayObject* nrotations_atom2_py;
  PyArrayObject* mappings_atom2_py;
  PyArrayObject* rotations_atom3_py;
  PyArrayObject* nrotations_atom3_py;
  PyArrayObject* mappings_atom3_py;
  PyArrayObject* lattice_py;
  PyArrayObject* transformations_py;
  PyArrayObject* independents_py;

  double precision;
  int i, j, k;
  int trans_dimension[3], ind_dimension[1];
  F3ArbiLenDBL *transformations;
  int ntriplets, natoms, nind1, nind2, maxrot2, maxrot3;
  int *mappings_atom2_1d, *mappings_atom3_1d, *nrotations_atom2, *nrotations_atom3;
  int (*rotations_atom2)[3][3], (*rotations_atom3)[3][3], *independents;
  double (*lattice)[3];
  VecDBL *positions;
  Triplet *triplets;
  Symmetry *symmetries_atom1;
  PointSymmetry *ps_atom2, *ps_atom3;
  VecArbiLenINT *mappings_atom1;
  MatArbiLenINT *mappings_atom2;
  F3ArbiLenINT *mappings_atom3;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOd",
      &triplets_reduced_py,
      &positions_py,
      &rotations_atom1_py,
      &translations_atom1_py,
      &mappings_atom1_py,
      &rotations_atom2_py,
      &nrotations_atom2_py,
      &mappings_atom2_py,
      &rotations_atom3_py,
      &nrotations_atom3_py,
      &mappings_atom3_py,
      &lattice_py,
      &precision)) {
    return NULL;
  }
  
  ntriplets = (int)triplets_reduced_py->dimensions[0];
  natoms = (int)positions_py->dimensions[0];
  nind1 = (int)mappings_atom3_py->dimensions[0];
  nind2 = (int)mappings_atom3_py->dimensions[1];
  mappings_atom2_1d = (int*) mappings_atom2_py->data;
  mappings_atom3_1d = (int*) mappings_atom3_py->data;
  
  rotations_atom2 = (int (*)[3][3]) rotations_atom2_py->data;
  nrotations_atom2 = (int *) nrotations_atom2_py->data;
  maxrot2 = (int)rotations_atom2_py->dimensions[1];
  rotations_atom3 = (int (*)[3][3]) rotations_atom3_py->data;
  nrotations_atom3 = (int *) nrotations_atom3_py->data;
  maxrot3 = (int)rotations_atom3_py->dimensions[2];
  
  lattice = (double (*)[3])lattice_py->data;
  independents = ivector(ntriplets * 27);
  positions = (VecDBL*)malloc(sizeof(VecDBL));
  triplets = (Triplet*)malloc(sizeof(Triplet));
  symmetries_atom1 = (Symmetry*)malloc(sizeof(Symmetry));
  ps_atom2 = (PointSymmetry*)malloc(sizeof(PointSymmetry) * nind1);
  ps_atom3 = (PointSymmetry*)malloc(sizeof(PointSymmetry) * nind1 * nind2);
  mappings_atom1 = (VecArbiLenINT*) malloc(sizeof(VecArbiLenINT));
  mappings_atom2 = alloc_MatArbiLenINT(nind1, natoms);
  mappings_atom3 = alloc_F3ArbiLenINT(nind1, nind2, natoms);
  
  triplets->size = (int)triplets_reduced_py->dimensions[0];
  triplets->tri = (int(*)[3])triplets_reduced_py->data;
  positions->size = (int)positions_py->dimensions[0];
  positions->vec = (double (*)[3]) positions_py->data;
  symmetries_atom1->size = (int)rotations_atom1_py->dimensions[0];
  symmetries_atom1->rot = (int (*)[3][3]) rotations_atom1_py->data;
  symmetries_atom1->trans = (double (*)[3]) translations_atom1_py->data;
  for (i=0; i<nind1; i++)
  {
    (ps_atom2+i)->size = nrotations_atom2[i];
    (ps_atom2+i)->rot = rotations_atom2 + i * maxrot2;
  }
  
  mappings_atom1->n = natoms;
  mappings_atom1->vec = (int *) mappings_atom1_py->data;
  for (i=0; i<nind1; i++)
  {
    for (j=0; j<nind2; j++)
    {
      (ps_atom3+i*nind2+j)->size = nrotations_atom3[i *nind2 + j];
      (ps_atom3+i*nind2+j)->rot = rotations_atom3 + i * nind2 * maxrot3 + j * maxrot3;
    }
  }
  
  for (i=0; i<nind1; i++)
    for (j=0; j<natoms; j++)
      mappings_atom2->mat[i][j] = mappings_atom2_1d[i*natoms+j];

  for (i=0; i<nind1; i++)
    for (j=0; j<nind2; j++)
      for (k=0; k<natoms; k++)
        mappings_atom3->f3[i][j][k] = mappings_atom3_1d[i*nind2*natoms+j*natoms+k];
  
  transformations = get_fc3_spg_invariance(independents,
                      triplets,
                      positions,
                      symmetries_atom1,
                      mappings_atom1,
                      ps_atom2,
                      mappings_atom2,
                      ps_atom3,
                      mappings_atom3,
                      lattice,
                      precision);
  trans_dimension[0] = ntriplets; trans_dimension[1] = 27; trans_dimension[2] = transformations->depth;
  ind_dimension[0] = transformations->depth;
  independents_py = (PyArrayObject*) PyArray_FromDims(1, ind_dimension, PyArray_INT);
  transformations_py = (PyArrayObject*) PyArray_FromDims(3, trans_dimension, PyArray_DOUBLE);
  
  if (independents_py == NULL ||  transformations_py == NULL) {
    warning_print("(PyArray_FromDims, line %d, %s).\n", __LINE__, __FILE__);
    runerror("Memory could not be allocated ");
    exit(1);
    }
    
  for (i=0; i<trans_dimension[0]; i++)
    for (j=0; j<trans_dimension[1]; j++)
      for (k=0; k<trans_dimension[2]; k++)
      {
        *((double*)transformations_py->data + 
          i * trans_dimension[1] * trans_dimension[2] + 
          j * trans_dimension[2] + k) = transformations->f3[i][j][k];
      }
      
  for (i=0; i<ind_dimension[0]; i++)
  {
    *((int*)independents_py->data + i) = independents[i];
  }
  free(positions);
  free(triplets);
  free(symmetries_atom1);
  free(ps_atom2);
  free(ps_atom3);
  free(mappings_atom1);
  free_F3ArbiLenDBL(transformations);
  free_ivector(independents);
  free_MatArbiLenINT(mappings_atom2);
  free_F3ArbiLenINT(mappings_atom3);
  tuple_return = Py_BuildValue("(OO)",PyArray_Return(independents_py), PyArray_Return(transformations_py));
  return tuple_return;
}


static PyObject * py_test(PyObject *self, PyObject *args)
{
  Py_Initialize();
  import_array();
  PyObject *tuple_return;
  PyArrayObject *positions_py;
  PyArrayObject *lattice_py;
  PyArrayObject* positions_return;
  PyArrayObject* lattice_return;

  int i, j, nd=2;
  double (*positions)[3];
  double (*lattice)[3];
  double (*temp)[3];
  int dimp[2], diml[2];
  diml[0] =3; diml[1]=3;
  if (!PyArg_ParseTuple(args, "OO",
      &positions_py,
      &lattice_py)) {
    return NULL;
  }
  
  dimp[0] = (int)positions_py->dimensions[0]; dimp[1] = 3;

  lattice = (double (*)[3])lattice_py->data;
  positions = (double (*)[3])positions_py->data;
  printf("log1\n");
  positions_return = (PyArrayObject*) PyArray_FromDims(nd, dimp, NPY_DOUBLE);
  lattice_return = (PyArrayObject*) PyArray_FromDims(nd, diml, NPY_DOUBLE);
  printf("log2\n");
  temp = (double (*)[3])positions_return->data;
  for (i=0; i<dimp[0]; i++)
    for (j=0; j<dimp[1]; j++)
    temp[i][j] = positions[i][j];
  
  for (i=0; i<diml[0]; i++)
    for (j=0; j<diml[1]; j++)
    {
      *((double*)lattice_return->data +
      i * diml[1] +j) = lattice[i][j];
    }
  tuple_return = Py_BuildValue("(OO)",lattice_return, positions_return);
  return tuple_return;
}

static PyObject * py_gaussian(PyObject *self, PyObject *args)
{
  Py_Initialize();
  import_array();
  PyArrayObject* matrix_py;
  PyArrayObject* transform_py;
  PyArrayObject* independents_py;
  MatArbiLenDBL* matrix, *transform;
  double *matrix1D, *transform1D, precision;
  int *independents;
  int row, column, i, j, num_independent;
  if (!PyArg_ParseTuple(args, "OOOd",
      &transform_py,
      &matrix_py,
      &independents_py,
      &precision)) {
    return NULL;
  }
  row = (int)matrix_py->dimensions[0];
  column = (int) matrix_py->dimensions[1];
  matrix1D = (double*) matrix_py->data;
  transform1D = (double*)transform_py->data;
  independents = (int*)independents_py->data;
  matrix = alloc_MatArbiLenDBL(row, column);
  transform = alloc_MatArbiLenDBL(column, column);
  for (i = 0; i < row; i++)
    for (j = 0; j < column; j++)
      matrix->mat[i][j] = matrix1D[i * column + j];
  num_independent = gaussian(transform->mat, independents, matrix->mat, row, column, precision);
  for (i = 0; i < column; i++)
    for (j = 0; j < num_independent; j++)
      transform1D[i * column + j] = transform->mat[i][j];
  free_MatArbiLenINT(matrix);
  free_MatArbiLenINT(transform);
  return PyInt_FromLong((long) num_independent);
}

static PyObject * py_rearrange_disp_fc3(PyObject *self, PyObject *args)
{
  PyArrayObject *py_ddcs, *py_disps, *py_coeff, *py_trans, *py_ifcmap;
  double coeff_cutoff;
  if (!PyArg_ParseTuple(args, "OOOOOd",
      &py_ddcs,
      &py_disps,
      &py_coeff,
      &py_trans,
      &py_ifcmap, 
      &coeff_cutoff)) {
    return NULL;
  }
  double *ddcs = (double*)py_ddcs->data;
  const double *disps = (double*) py_disps->data;
  const double *coeff = (double*) py_coeff->data;
  const double *trans = (double*) py_trans->data;
  const int *ifcmap = (int*) py_ifcmap->data;
  const int num_irred = py_trans->dimensions[2];
  const int num_step = py_disps->dimensions[0];
  const int num_atom = py_disps->dimensions[1];
  rearrange_disp_fc3(ddcs, disps, coeff, trans, ifcmap, num_step, num_atom, num_irred, coeff_cutoff);
  Py_RETURN_NONE;
}

static PyObject * py_rearrange_disp_fc2(PyObject *self, PyObject *args)
{
  PyArrayObject *py_ddcs, *py_disps, *py_coeff, *py_trans, *py_ifcmap;
  double coeff_cutoff;
  if (!PyArg_ParseTuple(args, "OOOOOd",
      &py_ddcs,
      &py_disps,
      &py_coeff,
      &py_trans,
      &py_ifcmap, 
      &coeff_cutoff)) {
    return NULL;
  }
  double *ddcs = (double*)py_ddcs->data;
  const double *disps = (double*) py_disps->data;
  const double *coeff = (double*) py_coeff->data;
  const double *trans = (double*) py_trans->data;
  const int *ifcmap = (int*) py_ifcmap->data;
  const int num_irred = py_trans->dimensions[2];
  const int num_step = py_disps->dimensions[0];
  const int num_atom = py_disps->dimensions[1];
  rearrange_disp_fc2(ddcs, disps, coeff, trans, ifcmap, num_step, num_atom, num_irred, coeff_cutoff);
  Py_RETURN_NONE;
}

static PyObject * py_get_fc3_coefficients(PyObject *self, PyObject *args)
{
  Py_Initialize();
  import_array();
  PyArrayObject* coeff_py;
  PyArrayObject* ifcmap3_py;
  PyArrayObject* triplets_py;
  PyArrayObject* first_atoms_py;
  PyArrayObject* triplets_mapping_py;
  PyArrayObject* triplets_transform_py;
  PyArrayObject* positions_py;
  PyArrayObject* rotations_atom1_py;
  PyArrayObject* translations_atom1_py;
  PyArrayObject* mappings_atom1_py;
  PyArrayObject* mapope_atom1_py;
  PyArrayObject* rotations_atom2_py;
  PyArrayObject* mappings_atom2_py;
  PyArrayObject* mapope_atom2_py;
  PyArrayObject* rotations_atom3_py;
  PyArrayObject* mapope_atom3_py;
  PyArrayObject* mappings_atom3_py;
  PyArrayObject* lattice_py;
  double precision;

  int i, j, k;
  int ntriplets, natoms, nind1, nind2, maxrot2, maxrot3;
  int *mappings_atom2_1d, *mappings_atom3_1d, *mapope_atom2_1d, *mapope_atom3_1d;
  int (*rotations_atom2)[3][3], (*rotations_atom3)[3][3];
  double (*lattice)[3];
  double (*coefficients)[27][27];
  double (*triplets_transform)[27][27];
  int *ifcmap3;
  int *triplets_mapping;
  VecDBL *positions;
  Triplet *triplets;
  Symmetry *symmetries_atom1;
  PointSymmetry *ps_atom2, *ps_atom3;
  VecArbiLenINT *mappings_atom1;
  VecArbiLenINT *mapope_atom1;
  VecArbiLenINT *first_atoms;
  MatArbiLenINT *mappings_atom2;
  MatArbiLenINT *mapope_atom2;
  F3ArbiLenINT *mappings_atom3;
  F3ArbiLenINT *mapope_atom3;
  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOOOOd",
      &coeff_py,
      &ifcmap3_py,
      &first_atoms_py,
      &triplets_py,
      &triplets_mapping_py,
      &triplets_transform_py,
      &lattice_py,
      &positions_py,
      &rotations_atom1_py,
      &translations_atom1_py,
      &mappings_atom1_py,
      &mapope_atom1_py,
      &rotations_atom2_py,
      &mappings_atom2_py,
      &mapope_atom2_py,
      &rotations_atom3_py,
      &mappings_atom3_py,
      &mapope_atom3_py,
      &precision)) {
    return NULL;
  }
  coefficients = (double (*)[27][27])coeff_py->data;
  ifcmap3 = (int*)ifcmap3_py->data;
  ntriplets = (int)triplets_py->dimensions[0];
  triplets_mapping = (int*)triplets_mapping_py->data;
  triplets_transform = (double (*)[27][27])triplets_transform_py->data;
  natoms = (int)positions_py->dimensions[0];
  nind1 = (int)mappings_atom3_py->dimensions[0];
  nind2 = (int)mappings_atom3_py->dimensions[1];
  mappings_atom2_1d = (int*) mappings_atom2_py->data;
  mapope_atom2_1d = (int*) mapope_atom2_py->data;
  mappings_atom3_1d = (int*) mappings_atom3_py->data;
  mapope_atom3_1d = (int*) mapope_atom3_py->data;
  rotations_atom2 = (int (*)[3][3]) rotations_atom2_py->data;
  maxrot2 = (int)rotations_atom2_py->dimensions[1];
  rotations_atom3 = (int (*)[3][3]) rotations_atom3_py->data;
  maxrot3 = (int)rotations_atom3_py->dimensions[2];
  lattice = (double (*)[3])lattice_py->data;
  positions = (VecDBL*)malloc(sizeof(VecDBL));
  triplets = (Triplet*)malloc(sizeof(Triplet));
  symmetries_atom1 = (Symmetry*)malloc(sizeof(Symmetry));
  ps_atom2 = (PointSymmetry*)malloc(sizeof(PointSymmetry) * nind1);
  ps_atom3 = (PointSymmetry*)malloc(sizeof(PointSymmetry) * nind1 * nind2);
  mappings_atom1 = (VecArbiLenINT*) malloc(sizeof(VecArbiLenINT));
  first_atoms = (VecArbiLenINT*) malloc(sizeof(VecArbiLenINT));
  mapope_atom1 = (VecArbiLenINT*) malloc(sizeof(VecArbiLenINT));
  mappings_atom2 = alloc_MatArbiLenINT(nind1, natoms);
  mapope_atom2 = alloc_MatArbiLenINT(nind1, natoms);
  mappings_atom3 = alloc_F3ArbiLenINT(nind1, nind2, natoms);
  mapope_atom3 = alloc_F3ArbiLenINT(nind1, nind2, natoms);
  mappings_atom1->vec = (int*) mappings_atom1_py->data;
  
  triplets->size = (int)triplets_py->dimensions[0];
  triplets->tri = (int(*)[3])triplets_py->data;
  positions->size = (int)positions_py->dimensions[0];
  positions->vec = (double (*)[3]) positions_py->data;
  symmetries_atom1->size = (int)rotations_atom1_py->dimensions[0];
  symmetries_atom1->rot = (int (*)[3][3]) rotations_atom1_py->data;
  symmetries_atom1->trans = (double (*)[3]) translations_atom1_py->data;
  for (i=0; i<nind1; i++)
  {
    (ps_atom2+i)->rot = rotations_atom2 + i * maxrot2;
  }
  
  mappings_atom1->n = natoms;
  mappings_atom1->vec = (int *) mappings_atom1_py->data;
  mapope_atom1->n = natoms;
  mapope_atom1->vec = (int*)mapope_atom1_py->data;
  first_atoms->n = first_atoms_py->dimensions[0];
  first_atoms->vec = (int*) first_atoms_py->data;

  for (i=0; i<nind1; i++)
  {
    for (j=0; j<nind2; j++)
    {
      (ps_atom3+i*nind2+j)->rot = rotations_atom3 + i * nind2 * maxrot3 + j * maxrot3;
    }
  }
  
  for (i=0; i<nind1; i++)
    for (j=0; j<natoms; j++)
    {
      mappings_atom2->mat[i][j] = mappings_atom2_1d[i*natoms+j];
      mapope_atom2->mat[i][j] = mapope_atom2_1d[i*natoms+j];
    }

  for (i=0; i<nind1; i++)
    for (j=0; j<nind2; j++)
      for (k=0; k<natoms; k++)
      {
        mappings_atom3->f3[i][j][k] = mappings_atom3_1d[i*nind2*natoms+j*natoms+k];
        mapope_atom3->f3[i][j][k] = mapope_atom3_1d[i*nind2*natoms+j*natoms+k];
      }
  get_fc3_coefficients(coefficients, 
                       ifcmap3,
                       first_atoms,
                       triplets,
                       triplets_mapping, 
                       triplets_transform,
                       lattice, 
                       positions, 
                       symmetries_atom1, 
                       mappings_atom1, 
                       mapope_atom1, 
                       ps_atom2,
                       mappings_atom2,
                       mapope_atom2,
                       ps_atom3, 
                       mappings_atom3, 
                       mapope_atom3, 
                       precision);
  free(positions);
  free(triplets);
  free(symmetries_atom1);
  free(ps_atom2);
  free(ps_atom3);
  free(mappings_atom1);
  free(mapope_atom1);
  free(first_atoms);
  free_MatArbiLenINT(mappings_atom2);
  free_MatArbiLenINT(mapope_atom2);
  free_F3ArbiLenINT(mappings_atom3);
  free_F3ArbiLenINT(mapope_atom3);
  Py_RETURN_NONE;
}

static PyObject * py_get_fc3_coefficients_triplet(PyObject *self, PyObject *args)
{
  Py_Initialize();
  import_array();
  PyArrayObject* coeff_py;
  PyArrayObject* triplet_py;
  PyArrayObject* triplets_py;
  PyArrayObject* triplets_mapping_py;
  PyArrayObject* triplets_transform_py;
  PyArrayObject* positions_py;
  PyArrayObject* rotations_atom1_py;
  PyArrayObject* translations_atom1_py;
  PyArrayObject* mappings_atom1_py;
  PyArrayObject* mapope_atom1_py;
  PyArrayObject* rotations_atom2_py;
  PyArrayObject* mappings_atom2_py;
  PyArrayObject* mapope_atom2_py;
  PyArrayObject* rotations_atom3_py;
  PyArrayObject* mapope_atom3_py;
  PyArrayObject* mappings_atom3_py;
  PyArrayObject* lattice_py;
  double precision;

  int i, j, k;
  int ntriplets, natoms, nind1, nind2, maxrot2, maxrot3;
  int *mappings_atom2_1d, *mappings_atom3_1d, *mapope_atom2_1d, *mapope_atom3_1d;
  int (*rotations_atom2)[3][3], (*rotations_atom3)[3][3];
  double (*lattice)[3];
  double (*coefficients)[27][27];
  double (*triplets_transform)[27][27];
  int *triplet;
  int *triplets_mapping;
  int ifc_map;
  VecDBL *positions;
  Triplet *triplets;
  Symmetry *symmetries_atom1;
  PointSymmetry *ps_atom2, *ps_atom3;
  VecArbiLenINT *mappings_atom1;
  VecArbiLenINT *mapope_atom1;
  MatArbiLenINT *mappings_atom2;
  MatArbiLenINT *mapope_atom2;
  F3ArbiLenINT *mappings_atom3;
  F3ArbiLenINT *mapope_atom3;
  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOOOd",
      &coeff_py,
      &triplet_py,
      &triplets_py,
      &triplets_mapping_py,
      &triplets_transform_py,
      &lattice_py,
      &positions_py,
      &rotations_atom1_py,
      &translations_atom1_py,
      &mappings_atom1_py,
      &mapope_atom1_py,
      &rotations_atom2_py,
      &mappings_atom2_py,
      &mapope_atom2_py,
      &rotations_atom3_py,
      &mappings_atom3_py,
      &mapope_atom3_py,
      &precision)) {
    return NULL;
  }
  coefficients = (double (*)[27])coeff_py->data;
  triplet = (int*)triplet_py->data;
  ntriplets = (int)triplets_py->dimensions[0];
  triplets_mapping = (int*)triplets_mapping_py->data;
  triplets_transform = (double (*)[27][27])triplets_transform_py->data;
  natoms = (int)positions_py->dimensions[0];
  nind1 = (int)mappings_atom3_py->dimensions[0];
  nind2 = (int)mappings_atom3_py->dimensions[1];
  mappings_atom2_1d = (int*) mappings_atom2_py->data;
  mapope_atom2_1d = (int*) mapope_atom2_py->data;
  mappings_atom3_1d = (int*) mappings_atom3_py->data;
  mapope_atom3_1d = (int*) mapope_atom3_py->data;
  rotations_atom2 = (int (*)[3][3]) rotations_atom2_py->data;
  maxrot2 = (int)rotations_atom2_py->dimensions[1];
  rotations_atom3 = (int (*)[3][3]) rotations_atom3_py->data;
  maxrot3 = (int)rotations_atom3_py->dimensions[2];
  lattice = (double (*)[3])lattice_py->data;
  positions = (VecDBL*)malloc(sizeof(VecDBL));
  triplets = (Triplet*)malloc(sizeof(Triplet));
  symmetries_atom1 = (Symmetry*)malloc(sizeof(Symmetry));
  ps_atom2 = (PointSymmetry*)malloc(sizeof(PointSymmetry) * nind1);
  ps_atom3 = (PointSymmetry*)malloc(sizeof(PointSymmetry) * nind1 * nind2);
  mappings_atom1 = (VecArbiLenINT*) malloc(sizeof(VecArbiLenINT));
  mapope_atom1 = (VecArbiLenINT*) malloc(sizeof(VecArbiLenINT));
  mappings_atom2 = alloc_MatArbiLenINT(nind1, natoms);
  mapope_atom2 = alloc_MatArbiLenINT(nind1, natoms);
  mappings_atom3 = alloc_F3ArbiLenINT(nind1, nind2, natoms);
  mapope_atom3 = alloc_F3ArbiLenINT(nind1, nind2, natoms);
  mappings_atom1->vec = (int*) mappings_atom1_py->data;

  triplets->size = (int)triplets_py->dimensions[0];
  triplets->tri = (int(*)[3])triplets_py->data;
  positions->size = (int)positions_py->dimensions[0];
  positions->vec = (double (*)[3]) positions_py->data;
  symmetries_atom1->size = (int)rotations_atom1_py->dimensions[0];
  symmetries_atom1->rot = (int (*)[3][3]) rotations_atom1_py->data;
  symmetries_atom1->trans = (double (*)[3]) translations_atom1_py->data;
  for (i=0; i<nind1; i++)
  {
    (ps_atom2+i)->rot = rotations_atom2 + i * maxrot2;
  }

  mappings_atom1->n = natoms;
  mappings_atom1->vec = (int *) mappings_atom1_py->data;
  mapope_atom1->n = natoms;
  mapope_atom1->vec = (int*)mapope_atom1_py->data;

  for (i=0; i<nind1; i++)
  {
    for (j=0; j<nind2; j++)
    {
      (ps_atom3+i*nind2+j)->rot = rotations_atom3 + i * nind2 * maxrot3 + j * maxrot3;
    }
  }

  for (i=0; i<nind1; i++)
    for (j=0; j<natoms; j++)
    {
      mappings_atom2->mat[i][j] = mappings_atom2_1d[i*natoms+j];
      mapope_atom2->mat[i][j] = mapope_atom2_1d[i*natoms+j];
    }

  for (i=0; i<nind1; i++)
    for (j=0; j<nind2; j++)
      for (k=0; k<natoms; k++)
      {
        mappings_atom3->f3[i][j][k] = mappings_atom3_1d[i*nind2*natoms+j*natoms+k];
        mapope_atom3->f3[i][j][k] = mapope_atom3_1d[i*nind2*natoms+j*natoms+k];
      }
  ifc_map = get_fc3_coefficients_triplet(coefficients,
                                       triplet,
                                       triplets,
                                       triplets_mapping,
                                       triplets_transform,
                                       lattice,
                                       positions,
                                       symmetries_atom1,
                                       mappings_atom1,
                                       mapope_atom1,
                                       ps_atom2,
                                       mappings_atom2,
                                       mapope_atom2,
                                       ps_atom3,
                                       mappings_atom3,
                                       mapope_atom3,
                                       precision);
  free(positions);
  free(triplets);
  free(symmetries_atom1);
  free(ps_atom2);
  free(ps_atom3);
  free(mappings_atom1);
  free(mapope_atom1);
  free_MatArbiLenINT(mappings_atom2);
  free_MatArbiLenINT(mapope_atom2);
  free_F3ArbiLenINT(mappings_atom3);
  free_F3ArbiLenINT(mapope_atom3);
  return PyInt_FromLong((long) ifc_map);
}