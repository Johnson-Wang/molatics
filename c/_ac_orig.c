#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include "ac_orig.h"
static PyObject * py_auto_correlation(PyObject *self, PyObject *args);
static PyObject * py_auto_correlation_all_atoms(PyObject *self, PyObject *args);
static PyMethodDef functions[] = {
  {"auto_correlation", py_auto_correlation, METH_VARARGS, "Calculate auto correlation for one atom"},
  {"auto_correlation_all_atoms", py_auto_correlation_all_atoms, METH_VARARGS, "Calculate auto correlation for all atoms"},
  {NULL, NULL, 0, NULL}
};
PyMODINIT_FUNC 
init_ac(void)
{
  Py_InitModule3("_ac", functions, "C-extension for autocorrelation\n\n...\n");
  return;
}

//autocorrelation for a variable with dimension[num_atom, num_correlation, sample_length, 3]
// the dimension of correlation is [num_atom, correlation_length, 3,3], which is a Cartesian tensor
static PyObject * py_auto_correlation_all_atoms(PyObject *self, PyObject *args)
{
  PyArrayObject* correlation;
  PyArrayObject* variable_all;

  if (!PyArg_ParseTuple(args, "OO",
			&correlation,
			&variable_all)) {
    return NULL;
  }
  
  double* corr = (double*)correlation->data;
  const double* variable = (double*)variable_all->data;
  const int num_atom = (int)correlation->dimensions[0];
  const int corr_length = (int)correlation->dimensions[1];
  const int num_corr = (int)variable_all->dimensions[1];
  const int sample_length = (int)variable_all->dimensions[2];

  auto_correlation_all_atoms(corr,variable,num_atom,num_corr, corr_length,sample_length);
  
  Py_RETURN_NONE;
}

//autocorrelation for a variable with dimension[num_correlation, sample_length, 3]
// the dimension of correlation is [correlation_length, 3,3], which is a Cartesian tensor
static PyObject * py_auto_correlation(PyObject *self, PyObject *args)
{
  PyArrayObject* correlation;
  PyArrayObject* variable_all;;

  if (!PyArg_ParseTuple(args, "OO",
			&correlation,
			&variable_all)) {
    return NULL;
  }
  
  double* corr = (double*)correlation->data;
  const double* variable = (double*)variable_all->data;
  const int corr_length = (int)correlation->dimensions[0];
  const int num_corr = (int)variable_all->dimensions[0];
  const int sample_length = (int)variable_all->dimensions[1];


  auto_correlation(corr,variable,num_corr, corr_length,sample_length);
  
  Py_RETURN_NONE;
}