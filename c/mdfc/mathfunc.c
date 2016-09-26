/* mathfunc.c */
/* Copyright (C) 2015 Wang Xinjiang */

#include <stdio.h>
#include <stdlib.h>
#include "mathfunc.h"
#define warning_print(...) fprintf(stderr, __VA_ARGS__)

void runerror(char error_text[])
{
  fprintf(stderr,"Run-time error...\n");
  fprintf(stderr,"%s\n",error_text);
  fprintf(stderr,"...now exiting to system...\n");
  exit(1);
}
MatINT * mat_alloc_MatINT(const int size)
{
  MatINT *matint;
  matint = (MatINT*) malloc( sizeof( MatINT ) );
  matint->size = size;
  if ( size > 0 ) {
    if ( ( matint->mat = (int (*)[3][3]) malloc( sizeof(int[3][3]) * size) )
   == NULL ) {
      warning_print("(MatINT, line %d, %s).\n", __LINE__, __FILE__);
    runerror("Memory could not be allocated ");
    }
  }
  return matint;
}

void mat_free_MatINT( MatINT * matint )
{
  if ( matint->size > 0 ) {
    free( matint->mat );
    matint->mat = NULL;
  }
  free( matint );
  matint = NULL;
}

VecDBL * mat_alloc_VecDBL( const int size )
{
  VecDBL *vecdbl;
  vecdbl = (VecDBL*) malloc( sizeof( VecDBL ) );
  vecdbl->size = size;
  if ( size > 0 ) {
    if ( ( vecdbl->vec = (double (*)[3]) malloc( sizeof(double[3]) * size) )
   == NULL ) {
      warning_print("(VecDBL, line %d, %s).\n", __LINE__, __FILE__);
    runerror("Memory could not be allocated ");
      exit(1);
    }
  }
  return vecdbl;
}

void mat_free_VecDBL( VecDBL * vecdbl )
{
  if ( vecdbl->size > 0 ) {
    free( vecdbl->vec );
    vecdbl->vec = NULL;
  }
  free( vecdbl );
  vecdbl = NULL;
}

Triplet * mat_alloc_Triplet(const int size)
{
  Triplet *triplet;
  triplet = (Triplet*) malloc( sizeof( Triplet ) );
  triplet->size = size;
  if ( size > 0 ) {
    if ( ( triplet->tri = (int (*)[3]) malloc( sizeof(int[3]) * size) )
   == NULL ) {
      warning_print("(Triplet, line %d, %s).\n", __LINE__, __FILE__);
    runerror("Memory could not be allocated ");
      exit(1);
    }
  }
  return triplet;
}
void mat_free_Triplet( Triplet * triplet )
{
  if ( triplet->size > 0 ) {
    free( triplet->tri );
    triplet->tri = NULL;
  }
  free( triplet );
  triplet = NULL;
}


int *ivector(long n)
/* allocate an int vector with subscript range v[0..n] */
{
  int *v;

  v=(int *)malloc((size_t) (n*sizeof(int)));
  if (!v) runerror("allocation failure in ivector()");
  return v;
}

double *dvector(long n)
/* allocate a double vector with subscript range v[nl..nh] */
{
  double *v;

  v=(double *)malloc((size_t) (n*sizeof(double)));
  if (!v) runerror("allocation failure in dvector()");
  return v;
}

double **dmatrix(long nr, long nc)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  long i;
  double **m;

  /* allocate pointers to rows */
  m=(double **) malloc((size_t)(nr*sizeof(double*)));
  if (!m) runerror("allocation failure 1 in matrix()");
  /* allocate rows and set pointers to them */
  m[0]=(double *) malloc((size_t)((nr*nc)*sizeof(double)));
  if (!m[0]) runerror("allocation failure 2 in matrix()");
  for(i=1;i<nr;i++) m[i]=m[i-1]+nc;

  /* return pointer to array of pointers to rows */
  return m;
}

int **imatrix(long nr, long nc)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  long i;
  int **m;

  /* allocate pointers to rows */
  m=(int **) malloc((size_t)(nr*sizeof(int*)));
  if (!m) runerror("allocation failure 1 in matrix()");
  /* allocate rows and set pointers to them */
  m[0]=(int *) malloc((size_t)((nr*nc)*sizeof(int)));
  if (!m[0]) runerror("allocation failure 2 in matrix()");
  for(i=1;i<nr;i++) m[i]=m[i-1]+nc;

  /* return pointer to array of pointers to rows */
  return m;
}

int ***if3tensor(long nr, long nc, long nd)
/* allocate a int 3tensor with range t[0..nr][0..nc][0..nd] */
{
  long i,j;
  int ***t;

  /* allocate pointers to pointers to rows */
  t=(int ***) malloc((size_t)((nr)*sizeof(int**)));
  if (!t) runerror("allocation failure 1 in f3tensor()");

  /* allocate pointers to rows and set pointers to them */
  t[0]=(int **) malloc((size_t)((nr*nc)*sizeof(int*)));
  if (!t[0]) runerror("allocation failure 2 in f3tensor()");

  /* allocate rows and set pointers to them */
  t[0][0]=(int *) malloc((size_t)((nr*nc*nd)*sizeof(int)));
  if (!t[0][0]) runerror("allocation failure 3 in f3tensor()");

  for(j=1;j<nc;j++) t[0][j]=t[0][j-1]+nd;
  for(i=1;i<nr;i++) {
    t[i]=t[i-1]+nc;
    t[i][0]=t[i-1][0]+nc*nd;
    for(j=1;j<nc;j++) t[i][j]=t[i][j-1]+nd;
  }
  /* return pointer to array of pointers to rows */
  return t;
}

double ***df3tensor(long nr, long nc, long nd)
/* allocate a double 3tensor with range t[0..nr][0..nc][0..nd] */
{
  long i,j;
  double ***t;

  /* allocate pointers to pointers to rows */
  t=(double ***) malloc((size_t)((nr)*sizeof(double**)));
  if (!t) runerror("allocation failure 1 in f3tensor()");

  /* allocate pointers to rows and set pointers to them */
  t[0]=(double **) malloc((size_t)((nr*nc)*sizeof(double*)));
  if (!t[0]) runerror("allocation failure 2 in f3tensor()");

  /* allocate rows and set pointers to them */
  t[0][0]=(double *) malloc((size_t)((nr*nc*nd)*sizeof(double)));
  if (!t[0][0]) runerror("allocation failure 3 in f3tensor()");

  for(j=1;j<nc;j++) t[0][j]=t[0][j-1]+nd;
  for(i=1;i<nr;i++) {
    t[i]=t[i-1]+nc;
    t[i][0]=t[i-1][0]+nc*nd;
    for(j=1;j<nc;j++) t[i][j]=t[i][j-1]+nd;
  }
  /* return pointer to array of pointers to rows */
  return t;
}


void free_ivector(int *v)
/* free an int vector allocated with ivector() */
{
  free((FREE_ARG) (v));
}


void free_dvector(double *v)
/* free a double vector allocated with dvector() */
{
  free((FREE_ARG) (v));
}


void free_dmatrix(double **m)
/* free a double matrix allocated by dmatrix() */
{
  free((FREE_ARG) (m[0]));
  free((FREE_ARG) (m));
}

void free_imatrix(int **m)
/* free an int matrix allocated by imatrix() */
{
  free((FREE_ARG) (m[0]));
  free((FREE_ARG) (m));
}

void free_if3tensor(int ***t)
/* free a int f3tensor allocated by f3tensor() */
{
  free((FREE_ARG) (t[0][0]));
  free((FREE_ARG) (t[0]));
  free((FREE_ARG) (t));
}

void free_df3tensor(double ***t)
/* free a double f3tensor allocated by f3tensor() */
{
  free((FREE_ARG) (t[0][0]));
  free((FREE_ARG) (t[0]));
  free((FREE_ARG) (t));
}

void init_ivector(int *v, const long length, int value)
/* init an int vector allocated with ivector() */
{
  int i;
  for (i=0; i<length; i++)
    v[i] = value;
}


void init_dvector(double *v, const long length, double value)
/* init a double vector allocated with dvector() */
{
  int i;
  for (i=0; i<length; i++)
    v[i] = value;
}


void init_dmatrix(double **m, const long row, const long column, double value)
/* init a double matrix allocated by dmatrix() */
{
  int i, j;
  for (i=0; i<row; i++)
    for (j=0; j<column; j++)
      m[i][j] = value;
}

void init_imatrix(int **m, const long row, const long column, int value)
/* init an int matrix allocated by imatrix() */
{
  int i, j;
  for (i=0; i<row; i++)
    for (j=0; j<column; j++)
      m[i][j] = value;
}

void init_if3tensor(int ***t, const long row, const long column, const long depth, int value)
/* init a int f3tensor allocated by f3tensor() */
{
  int i, j, k;
  for (i=0; i<row; i++)
    for (j=0; j<column; j++)
      for (k=0; k<depth; k++)
        t[i][j][k]=value;
}

void init_df3tensor(double ***t, const long row, const long column, const long depth, double value)
/* init a double f3tensor allocated by f3tensor() */
{
  int i, j, k;
  for (i=0; i<row; i++)
    for (j=0; j<column; j++)
      for (k=0; k<depth; k++)
        t[i][j][k]=value;
}

VecArbiLenINT *alloc_VecArbiLenINT(const int length)
{
  VecArbiLenINT *Vec;
  Vec = (VecArbiLenINT*) malloc(sizeof(VecArbiLenINT));
  Vec->n = length;
  if (length > 0){
    Vec->vec = ivector(length);
  }
  return Vec;
}

void free_VecArbiLenINT(VecArbiLenINT *Vec)
{
  if (Vec->n > 0){
    free_ivector(Vec->vec);
    Vec->vec = NULL;
  }
  free(Vec);
  Vec = NULL;
}

VecArbiLenDBL *alloc_VecArbiLenDBL(const int length)
{
  VecArbiLenDBL *Vec;
  Vec = (VecArbiLenDBL*) malloc(sizeof(VecArbiLenDBL));
  Vec->n = length;
  if (length > 0){
    Vec->vec = dvector(length);
  }
  return Vec;
}

void free_VecArbiLenDBL(VecArbiLenDBL *Vec)
{
  if (Vec->n > 0){
    free_dvector(Vec->vec);
    Vec->vec = NULL;
  }
  free(Vec);
  Vec = NULL;
} 

MatArbiLenINT *alloc_MatArbiLenINT(const int row, const int column)
{
  MatArbiLenINT *Mat;
  Mat = (MatArbiLenINT*) malloc(sizeof(MatArbiLenINT));
  Mat->row = row; Mat->column = column;
  if (row > 0 && column > 0){
    Mat->mat = imatrix(row, column);
  }
  return Mat;
}

void free_MatArbiLenINT(MatArbiLenINT *Mat)
{
  if (Mat->row > 0 && Mat->column > 0){
    free_imatrix(Mat->mat);
    Mat->mat = NULL;
  }
  free(Mat);
  Mat = NULL;
}

MatArbiLenDBL *alloc_MatArbiLenDBL(const int row, const int column)
{
  MatArbiLenDBL *Mat;
  Mat = (MatArbiLenDBL*) malloc(sizeof(MatArbiLenDBL));
  Mat->row = row; Mat->column = column;
  if (row > 0 && column > 0){
    Mat->mat = dmatrix(row, column);
  }
  return Mat;
}

void free_MatArbiLenDBL(MatArbiLenDBL *Mat)
{
  if (Mat->row > 0 && Mat->column > 0)
  {
    free_dmatrix(Mat->mat);
    Mat->mat = NULL;
  }
  free(Mat);
  Mat = NULL;
}

F3ArbiLenINT *alloc_F3ArbiLenINT(const int row, const int column, const int depth)
{
  F3ArbiLenINT *F3;
  F3 = (F3ArbiLenINT*) malloc(sizeof(F3ArbiLenINT));
  F3->row = row; F3->column = column; F3->depth = depth;
  if (row > 0 && column > 0 && depth > 0){
    F3->f3 = if3tensor(row, column, depth);
  }
  return F3;
}

void free_F3ArbiLenINT(F3ArbiLenINT *F3)
{
  if (F3->row > 0 && F3->column > 0 && F3->depth){
    free_if3tensor(F3->f3);
    F3->f3 = NULL;
  }
  free(F3);
  F3 = NULL;
}
F3ArbiLenDBL *alloc_F3ArbiLenDBL(const int row, const int column, const int depth)
{
  F3ArbiLenDBL *F3;
  F3 = (F3ArbiLenDBL*) malloc(sizeof(F3ArbiLenDBL));
  F3->row = row; F3->column = column; F3->depth = depth;
  if (row > 0 && column > 0 && depth > 0){
    F3->f3 = df3tensor(row, column, depth);
  }
  return F3;
}

void free_F3ArbiLenDBL(F3ArbiLenDBL *F3)
{
  if (F3->row > 0 && F3->column > 0 && F3->depth > 0){
    free_df3tensor(F3->f3);
    F3->f3 = NULL;
  }
  free(F3);
  F3 = NULL;
}

Symmetry * sym_alloc_symmetry(const int size)
{
  Symmetry *symmetry;

  symmetry = (Symmetry*) malloc(sizeof(Symmetry));
  symmetry->size = size;
  if (size > 0) {
    if ((symmetry->rot =
   (int (*)[3][3]) malloc(sizeof(int[3][3]) * size)) == NULL) {
      warning_print("(line %d, %s).\n", __LINE__, __FILE__);
      runerror("Memory could not be allocated ");
    }
    if ((symmetry->trans =
   (double (*)[3]) malloc(sizeof(double[3]) * size)) == NULL) {
      warning_print("(line %d, %s).\n", __LINE__, __FILE__);
      runerror("Memory could not be allocated ");
    }
  }
  return symmetry;
}

void sym_free_symmetry(Symmetry *symmetry)
{
  if (symmetry->size > 0) {
    free(symmetry->rot);
    symmetry->rot = NULL;
    free(symmetry->trans);
    symmetry->trans = NULL;
  }
  free(symmetry);
  symmetry = NULL;
}


PointSymmetry *sym_alloc_point_symmetry(const int size)
{
  PointSymmetry *ps;

  ps = (PointSymmetry*) malloc(sizeof(PointSymmetry));
  ps->size = size;
  if (size > 0) {
    if ((ps->rot =
   (int (*)[3][3]) malloc(sizeof(int[3][3]) * size)) == NULL) {
      warning_print("(line %d, %s).\n", __LINE__, __FILE__);
      runerror("Memory could not be allocated ");
    }
  }
  return ps;
}

void sym_free_point_symmetry(PointSymmetry * ps)
{
  if (ps->size > 0) {
    free(ps->rot);
    ps->rot = NULL;
  }
  free(ps);
  ps = NULL;
}
MatTensorINT * alloc_MatTensorINT( const int size )
{
  MatTensorINT *matt;

  matt = (MatTensorINT*) malloc(sizeof(MatTensorINT));
  matt->size = size;
  if (size > 0) {
    if ((matt->rot =
   (int (*)[27][27]) malloc(sizeof(int[27][27]) * size)) == NULL) {
      warning_print("(line %d, %s).\n", __LINE__, __FILE__);
      runerror("Memory could not be allocated ");
    }
  }
  return matt;
}

void free_MatTensorINT( MatTensorINT * matt )
{
  if (matt->size > 0) {
    free(matt->rot);
    matt->rot = NULL;
  }
  free(matt);
  matt = NULL;
}

MatTensorDBL * alloc_MatTensorDBL( const int size )
{
  MatTensorDBL *matt;

  matt = (MatTensorDBL*) malloc(sizeof(MatTensorDBL));
  matt->size = size;
  if (size > 0) {
    if ((matt->rot =
   (double (*)[27][27]) malloc(sizeof(double[27][27]) * size)) == NULL) {
      warning_print("(line %d, %s).\n", __LINE__, __FILE__);
      runerror("Memory could not be allocated ");
    }
  }
  return matt;
}

void free_MatTensorDBL( MatTensorDBL * matt )
{
  if (matt->size > 0) {
    free(matt->rot);
    matt->rot = NULL;
  }
  free(matt);
  matt = NULL;
}


double mat_get_determinant_d3(const double a[3][3])
{
  return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
    + a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2])
    + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

int mat_get_determinant_i3(const int a[3][3])
{
  return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
    + a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2])
    + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

int mat_get_trace_i3( const int a[3][3] )
{
  return a[0][0] + a[1][1] + a[2][2];
}

void mat_copy_array_i(int *a, const int *b, const int length)
{
  int i;
  for (i=0; i<length; i++)
  {
    a[i] = b[i];
  }
}
void mat_copy_matrix_d3(double a[3][3], const double b[3][3])
{
  a[0][0] = b[0][0];
  a[0][1] = b[0][1];
  a[0][2] = b[0][2];
  a[1][0] = b[1][0];
  a[1][1] = b[1][1];
  a[1][2] = b[1][2];
  a[2][0] = b[2][0];
  a[2][1] = b[2][1];
  a[2][2] = b[2][2];
}

void mat_copy_matrix_i3(int a[3][3], const int b[3][3])
{
  a[0][0] = b[0][0];
  a[0][1] = b[0][1];
  a[0][2] = b[0][2];
  a[1][0] = b[1][0];
  a[1][1] = b[1][1];
  a[1][2] = b[1][2];
  a[2][0] = b[2][0];
  a[2][1] = b[2][1];
  a[2][2] = b[2][2];
}

void mat_copy_vector_d3(double a[3], const double b[3])
{
  a[0] = b[0];
  a[1] = b[1];
  a[2] = b[2];
}

void mat_copy_vector_i3(int a[3], const int b[3])
{
  a[0] = b[0];
  a[1] = b[1];
  a[2] = b[2];
}
void mat_add_vector_d3(double c[3], const double a[3], const double b[3])
{
  c[0] = a[0] + b[0];
  c[1] = a[1] + b[1];
  c[2] = a[2] + b[2];
}

void mat_sub_vector_d3(double c[3], const double a[3], const double b[3])
{
  c[0] = a[0] - b[0];
  c[1] = a[1] - b[1];
  c[2] = a[2] - b[2];
}

int mat_check_int_vector_d3(const double a[3], const double prec)
{
  int  all_zero = 0;
  if (mat_Dabs(a[0] - mat_Nint(a[0])) < prec &&
      mat_Dabs(a[1] - mat_Nint(a[1])) < prec &&
      mat_Dabs(a[2] - mat_Nint(a[2])) < prec)
    all_zero = 1;
  return all_zero;
}

int mat_check_identity_matrix_i3(const int a[3][3],
         const int b[3][3])
{
  if ( a[0][0] - b[0][0] ||
       a[0][1] - b[0][1] ||
       a[0][2] - b[0][2] ||
       a[1][0] - b[1][0] ||
       a[1][1] - b[1][1] ||
       a[1][2] - b[1][2] ||
       a[2][0] - b[2][0] ||
       a[2][1] - b[2][1] ||
       a[2][2] - b[2][2]) {
    return 0;
  }
  else {
    return 1;
  }
}

int mat_check_identity_matrix_d3( const double a[3][3],
          const double b[3][3],
          const double symprec )
{
  if ( mat_Dabs( a[0][0] - b[0][0] ) > symprec ||
       mat_Dabs( a[0][1] - b[0][1] ) > symprec ||
       mat_Dabs( a[0][2] - b[0][2] ) > symprec ||
       mat_Dabs( a[1][0] - b[1][0] ) > symprec ||
       mat_Dabs( a[1][1] - b[1][1] ) > symprec ||
       mat_Dabs( a[1][2] - b[1][2] ) > symprec ||
       mat_Dabs( a[2][0] - b[2][0] ) > symprec ||
       mat_Dabs( a[2][1] - b[2][1] ) > symprec ||
       mat_Dabs( a[2][2] - b[2][2] ) > symprec ) {
    return 0;
  }
  else {
    return 1;
  }
}

int mat_check_identity_matrix_id3( const int a[3][3],
           const double b[3][3],
           const double symprec )
{
  if ( mat_Dabs( a[0][0] - b[0][0] ) > symprec ||
       mat_Dabs( a[0][1] - b[0][1] ) > symprec ||
       mat_Dabs( a[0][2] - b[0][2] ) > symprec ||
       mat_Dabs( a[1][0] - b[1][0] ) > symprec ||
       mat_Dabs( a[1][1] - b[1][1] ) > symprec ||
       mat_Dabs( a[1][2] - b[1][2] ) > symprec ||
       mat_Dabs( a[2][0] - b[2][0] ) > symprec ||
       mat_Dabs( a[2][1] - b[2][1] ) > symprec ||
       mat_Dabs( a[2][2] - b[2][2] ) > symprec ) {
    return 0;
  }
  else {
    return 1;
  }
}

/* m=axb */
void mat_kron_product_matrix3_d3(double Q[27][27],
                                const double a[3][3],
                                const double b[3][3],
                                const double c[3][3])
{
  int i, j, k, l, m, n;                   /* Q(ikm, jln) */
  for (i = 0; i < 3; i++) 
    for (j = 0; j < 3; j++) 
      for (k = 0; k < 3; k++)
        for (l = 0; l < 3; l++)
          for (m = 0; m < 3; m++)
            for (n = 0; n < 3; n++)
              Q[i * 9 + k * 3 + m][j * 9 + l * 3 + n] = a[i][j] * b[k][l] * c[m][n];
}

//void mat_kron_product_matrix3_permute_d3(double Q[27][27],
//          const double a[3][3],
//          const double b[3][3],
//          const double c[3][3],
//          const int permute[3])
//{ // Q(mik, jln) = a_ij * b_kl * c_mn
//  int i, j, k, l, m, n, index[3], p, q, r;
//  for (i=0; i<27; i++)
//    for (j=0; j<27; j++)
//      Q[i][j] = 0.;
//
//  for (i = 0; i < 3; i++)
//    for (j = 0; j < 3; j++)
//      for (k = 0; k < 3; k++)
//        for (l = 0; l < 3; l++)
//          for (m = 0; m < 3; m++)
//            for (n = 0; n < 3; n++)
//            {
//              index[0] = i; index[1] = k; index[2] = m;
//              p = index[permute[0]]; q = index[permute[1]]; r = index[permute[2]];
//              Q[p * 9 + q * 3 + r][j * 9 + l * 3 + n] = a[i][j] * b[k][l] * c[m][n];
//            }
//}

void mat_permute_d27(double PP[27][27], const int permute[3], const int axis)
{ // einstein expression (e.g.): ijk -->  jki
  // axis: 0 or 1, determining which axis to performe permutation on
  double Q[27][27];
  int i, j, k, l, m, n, ip, jp, kp, lp, mp, np, abc[3];
  for (i=0; i<27; i++)
    for (j=0; j<27; j++)
      Q[i][j] = 0.;

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      for (k = 0; k < 3; k++)
      {
        if (!axis){
          abc[0] = i; abc[1] = j; abc[2] = k;
          ip = abc[permute[0]]; jp = abc[permute[1]]; kp = abc[permute[2]];
        }
        else{
          ip = i; jp = j; kp = k;
        }
        for (l = 0; l < 3; l++)
          for (m = 0; m < 3; m++)
            for (n = 0; n < 3; n++)
            {
              if (axis){
                abc[0] = l; abc[1] = m; abc[2] = n;
                lp = abc[permute[0]]; mp = abc[permute[1]]; np = abc[permute[2]];
              }
              else{
                lp = l; mp = m; np = n;
              }
              Q[ip * 9 + jp * 3 + kp][lp * 9 + mp * 3 + np] = PP[i * 9 + j * 3 + k][l * 9 + m * 3 + n];
            }
      }
  mat_copy_mat_d27(PP, Q);
}

void mat_transpose_matrix_d27(double a[27][27])
{
  int i, j;
  double temp[27][27];
  for (i=0; i<27; i++)
    for (j=0; j<27; j++)
      temp[j][i] = a[i][j];
  mat_copy_mat_d27(a, temp);
}
void mat_copy_mat_i27(int a[27][27], const int b[27][27])
{
  a[ 0][ 0] = b[ 0][ 0];   a[ 0][ 1] = b[ 0][ 1];   a[ 0][ 2] = b[ 0][ 2];   a[ 0][ 3] = b[ 0][ 3];   a[ 0][ 4] = b[ 0][ 4];   a[ 0][ 5] = b[ 0][ 5];   a[ 0][ 6] = b[ 0][ 6];   a[ 0][ 7] = b[ 0][ 7];   a[ 0][ 8] = b[ 0][ 8];   a[ 0][ 9] = b[ 0][ 9];   a[ 0][10] = b[ 0][10];   a[ 0][11] = b[ 0][11];   a[ 0][12] = b[ 0][12];   a[ 0][13] = b[ 0][13];   a[ 0][14] = b[ 0][14];   a[ 0][15] = b[ 0][15];   a[ 0][16] = b[ 0][16];   a[ 0][17] = b[ 0][17];   a[ 0][18] = b[ 0][18];   a[ 0][19] = b[ 0][19];   a[ 0][20] = b[ 0][20];   a[ 0][21] = b[ 0][21];   a[ 0][22] = b[ 0][22];   a[ 0][23] = b[ 0][23];   a[ 0][24] = b[ 0][24];   a[ 0][25] = b[ 0][25];   a[ 0][26] = b[ 0][26];  
  a[ 1][ 0] = b[ 1][ 0];   a[ 1][ 1] = b[ 1][ 1];   a[ 1][ 2] = b[ 1][ 2];   a[ 1][ 3] = b[ 1][ 3];   a[ 1][ 4] = b[ 1][ 4];   a[ 1][ 5] = b[ 1][ 5];   a[ 1][ 6] = b[ 1][ 6];   a[ 1][ 7] = b[ 1][ 7];   a[ 1][ 8] = b[ 1][ 8];   a[ 1][ 9] = b[ 1][ 9];   a[ 1][10] = b[ 1][10];   a[ 1][11] = b[ 1][11];   a[ 1][12] = b[ 1][12];   a[ 1][13] = b[ 1][13];   a[ 1][14] = b[ 1][14];   a[ 1][15] = b[ 1][15];   a[ 1][16] = b[ 1][16];   a[ 1][17] = b[ 1][17];   a[ 1][18] = b[ 1][18];   a[ 1][19] = b[ 1][19];   a[ 1][20] = b[ 1][20];   a[ 1][21] = b[ 1][21];   a[ 1][22] = b[ 1][22];   a[ 1][23] = b[ 1][23];   a[ 1][24] = b[ 1][24];   a[ 1][25] = b[ 1][25];   a[ 1][26] = b[ 1][26];  
  a[ 2][ 0] = b[ 2][ 0];   a[ 2][ 1] = b[ 2][ 1];   a[ 2][ 2] = b[ 2][ 2];   a[ 2][ 3] = b[ 2][ 3];   a[ 2][ 4] = b[ 2][ 4];   a[ 2][ 5] = b[ 2][ 5];   a[ 2][ 6] = b[ 2][ 6];   a[ 2][ 7] = b[ 2][ 7];   a[ 2][ 8] = b[ 2][ 8];   a[ 2][ 9] = b[ 2][ 9];   a[ 2][10] = b[ 2][10];   a[ 2][11] = b[ 2][11];   a[ 2][12] = b[ 2][12];   a[ 2][13] = b[ 2][13];   a[ 2][14] = b[ 2][14];   a[ 2][15] = b[ 2][15];   a[ 2][16] = b[ 2][16];   a[ 2][17] = b[ 2][17];   a[ 2][18] = b[ 2][18];   a[ 2][19] = b[ 2][19];   a[ 2][20] = b[ 2][20];   a[ 2][21] = b[ 2][21];   a[ 2][22] = b[ 2][22];   a[ 2][23] = b[ 2][23];   a[ 2][24] = b[ 2][24];   a[ 2][25] = b[ 2][25];   a[ 2][26] = b[ 2][26];  
  a[ 3][ 0] = b[ 3][ 0];   a[ 3][ 1] = b[ 3][ 1];   a[ 3][ 2] = b[ 3][ 2];   a[ 3][ 3] = b[ 3][ 3];   a[ 3][ 4] = b[ 3][ 4];   a[ 3][ 5] = b[ 3][ 5];   a[ 3][ 6] = b[ 3][ 6];   a[ 3][ 7] = b[ 3][ 7];   a[ 3][ 8] = b[ 3][ 8];   a[ 3][ 9] = b[ 3][ 9];   a[ 3][10] = b[ 3][10];   a[ 3][11] = b[ 3][11];   a[ 3][12] = b[ 3][12];   a[ 3][13] = b[ 3][13];   a[ 3][14] = b[ 3][14];   a[ 3][15] = b[ 3][15];   a[ 3][16] = b[ 3][16];   a[ 3][17] = b[ 3][17];   a[ 3][18] = b[ 3][18];   a[ 3][19] = b[ 3][19];   a[ 3][20] = b[ 3][20];   a[ 3][21] = b[ 3][21];   a[ 3][22] = b[ 3][22];   a[ 3][23] = b[ 3][23];   a[ 3][24] = b[ 3][24];   a[ 3][25] = b[ 3][25];   a[ 3][26] = b[ 3][26];  
  a[ 4][ 0] = b[ 4][ 0];   a[ 4][ 1] = b[ 4][ 1];   a[ 4][ 2] = b[ 4][ 2];   a[ 4][ 3] = b[ 4][ 3];   a[ 4][ 4] = b[ 4][ 4];   a[ 4][ 5] = b[ 4][ 5];   a[ 4][ 6] = b[ 4][ 6];   a[ 4][ 7] = b[ 4][ 7];   a[ 4][ 8] = b[ 4][ 8];   a[ 4][ 9] = b[ 4][ 9];   a[ 4][10] = b[ 4][10];   a[ 4][11] = b[ 4][11];   a[ 4][12] = b[ 4][12];   a[ 4][13] = b[ 4][13];   a[ 4][14] = b[ 4][14];   a[ 4][15] = b[ 4][15];   a[ 4][16] = b[ 4][16];   a[ 4][17] = b[ 4][17];   a[ 4][18] = b[ 4][18];   a[ 4][19] = b[ 4][19];   a[ 4][20] = b[ 4][20];   a[ 4][21] = b[ 4][21];   a[ 4][22] = b[ 4][22];   a[ 4][23] = b[ 4][23];   a[ 4][24] = b[ 4][24];   a[ 4][25] = b[ 4][25];   a[ 4][26] = b[ 4][26];  
  a[ 5][ 0] = b[ 5][ 0];   a[ 5][ 1] = b[ 5][ 1];   a[ 5][ 2] = b[ 5][ 2];   a[ 5][ 3] = b[ 5][ 3];   a[ 5][ 4] = b[ 5][ 4];   a[ 5][ 5] = b[ 5][ 5];   a[ 5][ 6] = b[ 5][ 6];   a[ 5][ 7] = b[ 5][ 7];   a[ 5][ 8] = b[ 5][ 8];   a[ 5][ 9] = b[ 5][ 9];   a[ 5][10] = b[ 5][10];   a[ 5][11] = b[ 5][11];   a[ 5][12] = b[ 5][12];   a[ 5][13] = b[ 5][13];   a[ 5][14] = b[ 5][14];   a[ 5][15] = b[ 5][15];   a[ 5][16] = b[ 5][16];   a[ 5][17] = b[ 5][17];   a[ 5][18] = b[ 5][18];   a[ 5][19] = b[ 5][19];   a[ 5][20] = b[ 5][20];   a[ 5][21] = b[ 5][21];   a[ 5][22] = b[ 5][22];   a[ 5][23] = b[ 5][23];   a[ 5][24] = b[ 5][24];   a[ 5][25] = b[ 5][25];   a[ 5][26] = b[ 5][26];  
  a[ 6][ 0] = b[ 6][ 0];   a[ 6][ 1] = b[ 6][ 1];   a[ 6][ 2] = b[ 6][ 2];   a[ 6][ 3] = b[ 6][ 3];   a[ 6][ 4] = b[ 6][ 4];   a[ 6][ 5] = b[ 6][ 5];   a[ 6][ 6] = b[ 6][ 6];   a[ 6][ 7] = b[ 6][ 7];   a[ 6][ 8] = b[ 6][ 8];   a[ 6][ 9] = b[ 6][ 9];   a[ 6][10] = b[ 6][10];   a[ 6][11] = b[ 6][11];   a[ 6][12] = b[ 6][12];   a[ 6][13] = b[ 6][13];   a[ 6][14] = b[ 6][14];   a[ 6][15] = b[ 6][15];   a[ 6][16] = b[ 6][16];   a[ 6][17] = b[ 6][17];   a[ 6][18] = b[ 6][18];   a[ 6][19] = b[ 6][19];   a[ 6][20] = b[ 6][20];   a[ 6][21] = b[ 6][21];   a[ 6][22] = b[ 6][22];   a[ 6][23] = b[ 6][23];   a[ 6][24] = b[ 6][24];   a[ 6][25] = b[ 6][25];   a[ 6][26] = b[ 6][26];  
  a[ 7][ 0] = b[ 7][ 0];   a[ 7][ 1] = b[ 7][ 1];   a[ 7][ 2] = b[ 7][ 2];   a[ 7][ 3] = b[ 7][ 3];   a[ 7][ 4] = b[ 7][ 4];   a[ 7][ 5] = b[ 7][ 5];   a[ 7][ 6] = b[ 7][ 6];   a[ 7][ 7] = b[ 7][ 7];   a[ 7][ 8] = b[ 7][ 8];   a[ 7][ 9] = b[ 7][ 9];   a[ 7][10] = b[ 7][10];   a[ 7][11] = b[ 7][11];   a[ 7][12] = b[ 7][12];   a[ 7][13] = b[ 7][13];   a[ 7][14] = b[ 7][14];   a[ 7][15] = b[ 7][15];   a[ 7][16] = b[ 7][16];   a[ 7][17] = b[ 7][17];   a[ 7][18] = b[ 7][18];   a[ 7][19] = b[ 7][19];   a[ 7][20] = b[ 7][20];   a[ 7][21] = b[ 7][21];   a[ 7][22] = b[ 7][22];   a[ 7][23] = b[ 7][23];   a[ 7][24] = b[ 7][24];   a[ 7][25] = b[ 7][25];   a[ 7][26] = b[ 7][26];  
  a[ 8][ 0] = b[ 8][ 0];   a[ 8][ 1] = b[ 8][ 1];   a[ 8][ 2] = b[ 8][ 2];   a[ 8][ 3] = b[ 8][ 3];   a[ 8][ 4] = b[ 8][ 4];   a[ 8][ 5] = b[ 8][ 5];   a[ 8][ 6] = b[ 8][ 6];   a[ 8][ 7] = b[ 8][ 7];   a[ 8][ 8] = b[ 8][ 8];   a[ 8][ 9] = b[ 8][ 9];   a[ 8][10] = b[ 8][10];   a[ 8][11] = b[ 8][11];   a[ 8][12] = b[ 8][12];   a[ 8][13] = b[ 8][13];   a[ 8][14] = b[ 8][14];   a[ 8][15] = b[ 8][15];   a[ 8][16] = b[ 8][16];   a[ 8][17] = b[ 8][17];   a[ 8][18] = b[ 8][18];   a[ 8][19] = b[ 8][19];   a[ 8][20] = b[ 8][20];   a[ 8][21] = b[ 8][21];   a[ 8][22] = b[ 8][22];   a[ 8][23] = b[ 8][23];   a[ 8][24] = b[ 8][24];   a[ 8][25] = b[ 8][25];   a[ 8][26] = b[ 8][26];  
  a[ 9][ 0] = b[ 9][ 0];   a[ 9][ 1] = b[ 9][ 1];   a[ 9][ 2] = b[ 9][ 2];   a[ 9][ 3] = b[ 9][ 3];   a[ 9][ 4] = b[ 9][ 4];   a[ 9][ 5] = b[ 9][ 5];   a[ 9][ 6] = b[ 9][ 6];   a[ 9][ 7] = b[ 9][ 7];   a[ 9][ 8] = b[ 9][ 8];   a[ 9][ 9] = b[ 9][ 9];   a[ 9][10] = b[ 9][10];   a[ 9][11] = b[ 9][11];   a[ 9][12] = b[ 9][12];   a[ 9][13] = b[ 9][13];   a[ 9][14] = b[ 9][14];   a[ 9][15] = b[ 9][15];   a[ 9][16] = b[ 9][16];   a[ 9][17] = b[ 9][17];   a[ 9][18] = b[ 9][18];   a[ 9][19] = b[ 9][19];   a[ 9][20] = b[ 9][20];   a[ 9][21] = b[ 9][21];   a[ 9][22] = b[ 9][22];   a[ 9][23] = b[ 9][23];   a[ 9][24] = b[ 9][24];   a[ 9][25] = b[ 9][25];   a[ 9][26] = b[ 9][26];  
  a[10][ 0] = b[10][ 0];   a[10][ 1] = b[10][ 1];   a[10][ 2] = b[10][ 2];   a[10][ 3] = b[10][ 3];   a[10][ 4] = b[10][ 4];   a[10][ 5] = b[10][ 5];   a[10][ 6] = b[10][ 6];   a[10][ 7] = b[10][ 7];   a[10][ 8] = b[10][ 8];   a[10][ 9] = b[10][ 9];   a[10][10] = b[10][10];   a[10][11] = b[10][11];   a[10][12] = b[10][12];   a[10][13] = b[10][13];   a[10][14] = b[10][14];   a[10][15] = b[10][15];   a[10][16] = b[10][16];   a[10][17] = b[10][17];   a[10][18] = b[10][18];   a[10][19] = b[10][19];   a[10][20] = b[10][20];   a[10][21] = b[10][21];   a[10][22] = b[10][22];   a[10][23] = b[10][23];   a[10][24] = b[10][24];   a[10][25] = b[10][25];   a[10][26] = b[10][26];  
  a[11][ 0] = b[11][ 0];   a[11][ 1] = b[11][ 1];   a[11][ 2] = b[11][ 2];   a[11][ 3] = b[11][ 3];   a[11][ 4] = b[11][ 4];   a[11][ 5] = b[11][ 5];   a[11][ 6] = b[11][ 6];   a[11][ 7] = b[11][ 7];   a[11][ 8] = b[11][ 8];   a[11][ 9] = b[11][ 9];   a[11][10] = b[11][10];   a[11][11] = b[11][11];   a[11][12] = b[11][12];   a[11][13] = b[11][13];   a[11][14] = b[11][14];   a[11][15] = b[11][15];   a[11][16] = b[11][16];   a[11][17] = b[11][17];   a[11][18] = b[11][18];   a[11][19] = b[11][19];   a[11][20] = b[11][20];   a[11][21] = b[11][21];   a[11][22] = b[11][22];   a[11][23] = b[11][23];   a[11][24] = b[11][24];   a[11][25] = b[11][25];   a[11][26] = b[11][26];  
  a[12][ 0] = b[12][ 0];   a[12][ 1] = b[12][ 1];   a[12][ 2] = b[12][ 2];   a[12][ 3] = b[12][ 3];   a[12][ 4] = b[12][ 4];   a[12][ 5] = b[12][ 5];   a[12][ 6] = b[12][ 6];   a[12][ 7] = b[12][ 7];   a[12][ 8] = b[12][ 8];   a[12][ 9] = b[12][ 9];   a[12][10] = b[12][10];   a[12][11] = b[12][11];   a[12][12] = b[12][12];   a[12][13] = b[12][13];   a[12][14] = b[12][14];   a[12][15] = b[12][15];   a[12][16] = b[12][16];   a[12][17] = b[12][17];   a[12][18] = b[12][18];   a[12][19] = b[12][19];   a[12][20] = b[12][20];   a[12][21] = b[12][21];   a[12][22] = b[12][22];   a[12][23] = b[12][23];   a[12][24] = b[12][24];   a[12][25] = b[12][25];   a[12][26] = b[12][26];  
  a[13][ 0] = b[13][ 0];   a[13][ 1] = b[13][ 1];   a[13][ 2] = b[13][ 2];   a[13][ 3] = b[13][ 3];   a[13][ 4] = b[13][ 4];   a[13][ 5] = b[13][ 5];   a[13][ 6] = b[13][ 6];   a[13][ 7] = b[13][ 7];   a[13][ 8] = b[13][ 8];   a[13][ 9] = b[13][ 9];   a[13][10] = b[13][10];   a[13][11] = b[13][11];   a[13][12] = b[13][12];   a[13][13] = b[13][13];   a[13][14] = b[13][14];   a[13][15] = b[13][15];   a[13][16] = b[13][16];   a[13][17] = b[13][17];   a[13][18] = b[13][18];   a[13][19] = b[13][19];   a[13][20] = b[13][20];   a[13][21] = b[13][21];   a[13][22] = b[13][22];   a[13][23] = b[13][23];   a[13][24] = b[13][24];   a[13][25] = b[13][25];   a[13][26] = b[13][26];  
  a[14][ 0] = b[14][ 0];   a[14][ 1] = b[14][ 1];   a[14][ 2] = b[14][ 2];   a[14][ 3] = b[14][ 3];   a[14][ 4] = b[14][ 4];   a[14][ 5] = b[14][ 5];   a[14][ 6] = b[14][ 6];   a[14][ 7] = b[14][ 7];   a[14][ 8] = b[14][ 8];   a[14][ 9] = b[14][ 9];   a[14][10] = b[14][10];   a[14][11] = b[14][11];   a[14][12] = b[14][12];   a[14][13] = b[14][13];   a[14][14] = b[14][14];   a[14][15] = b[14][15];   a[14][16] = b[14][16];   a[14][17] = b[14][17];   a[14][18] = b[14][18];   a[14][19] = b[14][19];   a[14][20] = b[14][20];   a[14][21] = b[14][21];   a[14][22] = b[14][22];   a[14][23] = b[14][23];   a[14][24] = b[14][24];   a[14][25] = b[14][25];   a[14][26] = b[14][26];  
  a[15][ 0] = b[15][ 0];   a[15][ 1] = b[15][ 1];   a[15][ 2] = b[15][ 2];   a[15][ 3] = b[15][ 3];   a[15][ 4] = b[15][ 4];   a[15][ 5] = b[15][ 5];   a[15][ 6] = b[15][ 6];   a[15][ 7] = b[15][ 7];   a[15][ 8] = b[15][ 8];   a[15][ 9] = b[15][ 9];   a[15][10] = b[15][10];   a[15][11] = b[15][11];   a[15][12] = b[15][12];   a[15][13] = b[15][13];   a[15][14] = b[15][14];   a[15][15] = b[15][15];   a[15][16] = b[15][16];   a[15][17] = b[15][17];   a[15][18] = b[15][18];   a[15][19] = b[15][19];   a[15][20] = b[15][20];   a[15][21] = b[15][21];   a[15][22] = b[15][22];   a[15][23] = b[15][23];   a[15][24] = b[15][24];   a[15][25] = b[15][25];   a[15][26] = b[15][26];  
  a[16][ 0] = b[16][ 0];   a[16][ 1] = b[16][ 1];   a[16][ 2] = b[16][ 2];   a[16][ 3] = b[16][ 3];   a[16][ 4] = b[16][ 4];   a[16][ 5] = b[16][ 5];   a[16][ 6] = b[16][ 6];   a[16][ 7] = b[16][ 7];   a[16][ 8] = b[16][ 8];   a[16][ 9] = b[16][ 9];   a[16][10] = b[16][10];   a[16][11] = b[16][11];   a[16][12] = b[16][12];   a[16][13] = b[16][13];   a[16][14] = b[16][14];   a[16][15] = b[16][15];   a[16][16] = b[16][16];   a[16][17] = b[16][17];   a[16][18] = b[16][18];   a[16][19] = b[16][19];   a[16][20] = b[16][20];   a[16][21] = b[16][21];   a[16][22] = b[16][22];   a[16][23] = b[16][23];   a[16][24] = b[16][24];   a[16][25] = b[16][25];   a[16][26] = b[16][26];  
  a[17][ 0] = b[17][ 0];   a[17][ 1] = b[17][ 1];   a[17][ 2] = b[17][ 2];   a[17][ 3] = b[17][ 3];   a[17][ 4] = b[17][ 4];   a[17][ 5] = b[17][ 5];   a[17][ 6] = b[17][ 6];   a[17][ 7] = b[17][ 7];   a[17][ 8] = b[17][ 8];   a[17][ 9] = b[17][ 9];   a[17][10] = b[17][10];   a[17][11] = b[17][11];   a[17][12] = b[17][12];   a[17][13] = b[17][13];   a[17][14] = b[17][14];   a[17][15] = b[17][15];   a[17][16] = b[17][16];   a[17][17] = b[17][17];   a[17][18] = b[17][18];   a[17][19] = b[17][19];   a[17][20] = b[17][20];   a[17][21] = b[17][21];   a[17][22] = b[17][22];   a[17][23] = b[17][23];   a[17][24] = b[17][24];   a[17][25] = b[17][25];   a[17][26] = b[17][26];  
  a[18][ 0] = b[18][ 0];   a[18][ 1] = b[18][ 1];   a[18][ 2] = b[18][ 2];   a[18][ 3] = b[18][ 3];   a[18][ 4] = b[18][ 4];   a[18][ 5] = b[18][ 5];   a[18][ 6] = b[18][ 6];   a[18][ 7] = b[18][ 7];   a[18][ 8] = b[18][ 8];   a[18][ 9] = b[18][ 9];   a[18][10] = b[18][10];   a[18][11] = b[18][11];   a[18][12] = b[18][12];   a[18][13] = b[18][13];   a[18][14] = b[18][14];   a[18][15] = b[18][15];   a[18][16] = b[18][16];   a[18][17] = b[18][17];   a[18][18] = b[18][18];   a[18][19] = b[18][19];   a[18][20] = b[18][20];   a[18][21] = b[18][21];   a[18][22] = b[18][22];   a[18][23] = b[18][23];   a[18][24] = b[18][24];   a[18][25] = b[18][25];   a[18][26] = b[18][26];  
  a[19][ 0] = b[19][ 0];   a[19][ 1] = b[19][ 1];   a[19][ 2] = b[19][ 2];   a[19][ 3] = b[19][ 3];   a[19][ 4] = b[19][ 4];   a[19][ 5] = b[19][ 5];   a[19][ 6] = b[19][ 6];   a[19][ 7] = b[19][ 7];   a[19][ 8] = b[19][ 8];   a[19][ 9] = b[19][ 9];   a[19][10] = b[19][10];   a[19][11] = b[19][11];   a[19][12] = b[19][12];   a[19][13] = b[19][13];   a[19][14] = b[19][14];   a[19][15] = b[19][15];   a[19][16] = b[19][16];   a[19][17] = b[19][17];   a[19][18] = b[19][18];   a[19][19] = b[19][19];   a[19][20] = b[19][20];   a[19][21] = b[19][21];   a[19][22] = b[19][22];   a[19][23] = b[19][23];   a[19][24] = b[19][24];   a[19][25] = b[19][25];   a[19][26] = b[19][26];  
  a[20][ 0] = b[20][ 0];   a[20][ 1] = b[20][ 1];   a[20][ 2] = b[20][ 2];   a[20][ 3] = b[20][ 3];   a[20][ 4] = b[20][ 4];   a[20][ 5] = b[20][ 5];   a[20][ 6] = b[20][ 6];   a[20][ 7] = b[20][ 7];   a[20][ 8] = b[20][ 8];   a[20][ 9] = b[20][ 9];   a[20][10] = b[20][10];   a[20][11] = b[20][11];   a[20][12] = b[20][12];   a[20][13] = b[20][13];   a[20][14] = b[20][14];   a[20][15] = b[20][15];   a[20][16] = b[20][16];   a[20][17] = b[20][17];   a[20][18] = b[20][18];   a[20][19] = b[20][19];   a[20][20] = b[20][20];   a[20][21] = b[20][21];   a[20][22] = b[20][22];   a[20][23] = b[20][23];   a[20][24] = b[20][24];   a[20][25] = b[20][25];   a[20][26] = b[20][26];  
  a[21][ 0] = b[21][ 0];   a[21][ 1] = b[21][ 1];   a[21][ 2] = b[21][ 2];   a[21][ 3] = b[21][ 3];   a[21][ 4] = b[21][ 4];   a[21][ 5] = b[21][ 5];   a[21][ 6] = b[21][ 6];   a[21][ 7] = b[21][ 7];   a[21][ 8] = b[21][ 8];   a[21][ 9] = b[21][ 9];   a[21][10] = b[21][10];   a[21][11] = b[21][11];   a[21][12] = b[21][12];   a[21][13] = b[21][13];   a[21][14] = b[21][14];   a[21][15] = b[21][15];   a[21][16] = b[21][16];   a[21][17] = b[21][17];   a[21][18] = b[21][18];   a[21][19] = b[21][19];   a[21][20] = b[21][20];   a[21][21] = b[21][21];   a[21][22] = b[21][22];   a[21][23] = b[21][23];   a[21][24] = b[21][24];   a[21][25] = b[21][25];   a[21][26] = b[21][26];  
  a[22][ 0] = b[22][ 0];   a[22][ 1] = b[22][ 1];   a[22][ 2] = b[22][ 2];   a[22][ 3] = b[22][ 3];   a[22][ 4] = b[22][ 4];   a[22][ 5] = b[22][ 5];   a[22][ 6] = b[22][ 6];   a[22][ 7] = b[22][ 7];   a[22][ 8] = b[22][ 8];   a[22][ 9] = b[22][ 9];   a[22][10] = b[22][10];   a[22][11] = b[22][11];   a[22][12] = b[22][12];   a[22][13] = b[22][13];   a[22][14] = b[22][14];   a[22][15] = b[22][15];   a[22][16] = b[22][16];   a[22][17] = b[22][17];   a[22][18] = b[22][18];   a[22][19] = b[22][19];   a[22][20] = b[22][20];   a[22][21] = b[22][21];   a[22][22] = b[22][22];   a[22][23] = b[22][23];   a[22][24] = b[22][24];   a[22][25] = b[22][25];   a[22][26] = b[22][26];  
  a[23][ 0] = b[23][ 0];   a[23][ 1] = b[23][ 1];   a[23][ 2] = b[23][ 2];   a[23][ 3] = b[23][ 3];   a[23][ 4] = b[23][ 4];   a[23][ 5] = b[23][ 5];   a[23][ 6] = b[23][ 6];   a[23][ 7] = b[23][ 7];   a[23][ 8] = b[23][ 8];   a[23][ 9] = b[23][ 9];   a[23][10] = b[23][10];   a[23][11] = b[23][11];   a[23][12] = b[23][12];   a[23][13] = b[23][13];   a[23][14] = b[23][14];   a[23][15] = b[23][15];   a[23][16] = b[23][16];   a[23][17] = b[23][17];   a[23][18] = b[23][18];   a[23][19] = b[23][19];   a[23][20] = b[23][20];   a[23][21] = b[23][21];   a[23][22] = b[23][22];   a[23][23] = b[23][23];   a[23][24] = b[23][24];   a[23][25] = b[23][25];   a[23][26] = b[23][26];  
  a[24][ 0] = b[24][ 0];   a[24][ 1] = b[24][ 1];   a[24][ 2] = b[24][ 2];   a[24][ 3] = b[24][ 3];   a[24][ 4] = b[24][ 4];   a[24][ 5] = b[24][ 5];   a[24][ 6] = b[24][ 6];   a[24][ 7] = b[24][ 7];   a[24][ 8] = b[24][ 8];   a[24][ 9] = b[24][ 9];   a[24][10] = b[24][10];   a[24][11] = b[24][11];   a[24][12] = b[24][12];   a[24][13] = b[24][13];   a[24][14] = b[24][14];   a[24][15] = b[24][15];   a[24][16] = b[24][16];   a[24][17] = b[24][17];   a[24][18] = b[24][18];   a[24][19] = b[24][19];   a[24][20] = b[24][20];   a[24][21] = b[24][21];   a[24][22] = b[24][22];   a[24][23] = b[24][23];   a[24][24] = b[24][24];   a[24][25] = b[24][25];   a[24][26] = b[24][26];  
  a[25][ 0] = b[25][ 0];   a[25][ 1] = b[25][ 1];   a[25][ 2] = b[25][ 2];   a[25][ 3] = b[25][ 3];   a[25][ 4] = b[25][ 4];   a[25][ 5] = b[25][ 5];   a[25][ 6] = b[25][ 6];   a[25][ 7] = b[25][ 7];   a[25][ 8] = b[25][ 8];   a[25][ 9] = b[25][ 9];   a[25][10] = b[25][10];   a[25][11] = b[25][11];   a[25][12] = b[25][12];   a[25][13] = b[25][13];   a[25][14] = b[25][14];   a[25][15] = b[25][15];   a[25][16] = b[25][16];   a[25][17] = b[25][17];   a[25][18] = b[25][18];   a[25][19] = b[25][19];   a[25][20] = b[25][20];   a[25][21] = b[25][21];   a[25][22] = b[25][22];   a[25][23] = b[25][23];   a[25][24] = b[25][24];   a[25][25] = b[25][25];   a[25][26] = b[25][26];  
  a[26][ 0] = b[26][ 0];   a[26][ 1] = b[26][ 1];   a[26][ 2] = b[26][ 2];   a[26][ 3] = b[26][ 3];   a[26][ 4] = b[26][ 4];   a[26][ 5] = b[26][ 5];   a[26][ 6] = b[26][ 6];   a[26][ 7] = b[26][ 7];   a[26][ 8] = b[26][ 8];   a[26][ 9] = b[26][ 9];   a[26][10] = b[26][10];   a[26][11] = b[26][11];   a[26][12] = b[26][12];   a[26][13] = b[26][13];   a[26][14] = b[26][14];   a[26][15] = b[26][15];   a[26][16] = b[26][16];   a[26][17] = b[26][17];   a[26][18] = b[26][18];   a[26][19] = b[26][19];   a[26][20] = b[26][20];   a[26][21] = b[26][21];   a[26][22] = b[26][22];   a[26][23] = b[26][23];   a[26][24] = b[26][24];   a[26][25] = b[26][25];   a[26][26] = b[26][26];
}
void mat_multiply_matrix_i27(int m[27][27],const int a[27][27],const int b[27][27])
{
  int i, j;                   /* a_ij */
  int c[27][27];
  for (i = 0; i < 27; i++) {
    for (j = 0; j < 27; j++) {
      c[i][j] =
        a[i][ 0] * b[ 0][j] +  a[i][ 1] * b[ 1][j] +  a[i][ 2] * b[ 2][j] +  a[i][ 3] * b[ 3][j] +  a[i][ 4] * b[ 4][j] +  a[i][ 5] * b[ 5][j] +  a[i][ 6] * b[ 6][j] +  a[i][ 7] * b[ 7][j] +  a[i][ 8] * b[ 8][j] +  a[i][ 9] * b[ 9][j] +  a[i][10] * b[10][j] +  a[i][11] * b[11][j] +  a[i][12] * b[12][j] +  a[i][13] * b[13][j] +  a[i][14] * b[14][j] +  a[i][15] * b[15][j] +  a[i][16] * b[16][j] +  a[i][17] * b[17][j] +  a[i][18] * b[18][j] +  a[i][19] * b[19][j] +  a[i][20] * b[20][j] +  a[i][21] * b[21][j] +  a[i][22] * b[22][j] +  a[i][23] * b[23][j] +  a[i][24] * b[24][j] +  a[i][25] * b[25][j] +  a[i][26] * b[26][j];
    }
  }
  mat_copy_mat_i27(m, c);
}

void mat_copy_mat_d27(double a[27][27], const double b[27][27])
{
  a[ 0][ 0] = b[ 0][ 0];   a[ 0][ 1] = b[ 0][ 1];   a[ 0][ 2] = b[ 0][ 2];   a[ 0][ 3] = b[ 0][ 3];   a[ 0][ 4] = b[ 0][ 4];   a[ 0][ 5] = b[ 0][ 5];   a[ 0][ 6] = b[ 0][ 6];   a[ 0][ 7] = b[ 0][ 7];   a[ 0][ 8] = b[ 0][ 8];   a[ 0][ 9] = b[ 0][ 9];   a[ 0][10] = b[ 0][10];   a[ 0][11] = b[ 0][11];   a[ 0][12] = b[ 0][12];   a[ 0][13] = b[ 0][13];   a[ 0][14] = b[ 0][14];   a[ 0][15] = b[ 0][15];   a[ 0][16] = b[ 0][16];   a[ 0][17] = b[ 0][17];   a[ 0][18] = b[ 0][18];   a[ 0][19] = b[ 0][19];   a[ 0][20] = b[ 0][20];   a[ 0][21] = b[ 0][21];   a[ 0][22] = b[ 0][22];   a[ 0][23] = b[ 0][23];   a[ 0][24] = b[ 0][24];   a[ 0][25] = b[ 0][25];   a[ 0][26] = b[ 0][26];  
  a[ 1][ 0] = b[ 1][ 0];   a[ 1][ 1] = b[ 1][ 1];   a[ 1][ 2] = b[ 1][ 2];   a[ 1][ 3] = b[ 1][ 3];   a[ 1][ 4] = b[ 1][ 4];   a[ 1][ 5] = b[ 1][ 5];   a[ 1][ 6] = b[ 1][ 6];   a[ 1][ 7] = b[ 1][ 7];   a[ 1][ 8] = b[ 1][ 8];   a[ 1][ 9] = b[ 1][ 9];   a[ 1][10] = b[ 1][10];   a[ 1][11] = b[ 1][11];   a[ 1][12] = b[ 1][12];   a[ 1][13] = b[ 1][13];   a[ 1][14] = b[ 1][14];   a[ 1][15] = b[ 1][15];   a[ 1][16] = b[ 1][16];   a[ 1][17] = b[ 1][17];   a[ 1][18] = b[ 1][18];   a[ 1][19] = b[ 1][19];   a[ 1][20] = b[ 1][20];   a[ 1][21] = b[ 1][21];   a[ 1][22] = b[ 1][22];   a[ 1][23] = b[ 1][23];   a[ 1][24] = b[ 1][24];   a[ 1][25] = b[ 1][25];   a[ 1][26] = b[ 1][26];  
  a[ 2][ 0] = b[ 2][ 0];   a[ 2][ 1] = b[ 2][ 1];   a[ 2][ 2] = b[ 2][ 2];   a[ 2][ 3] = b[ 2][ 3];   a[ 2][ 4] = b[ 2][ 4];   a[ 2][ 5] = b[ 2][ 5];   a[ 2][ 6] = b[ 2][ 6];   a[ 2][ 7] = b[ 2][ 7];   a[ 2][ 8] = b[ 2][ 8];   a[ 2][ 9] = b[ 2][ 9];   a[ 2][10] = b[ 2][10];   a[ 2][11] = b[ 2][11];   a[ 2][12] = b[ 2][12];   a[ 2][13] = b[ 2][13];   a[ 2][14] = b[ 2][14];   a[ 2][15] = b[ 2][15];   a[ 2][16] = b[ 2][16];   a[ 2][17] = b[ 2][17];   a[ 2][18] = b[ 2][18];   a[ 2][19] = b[ 2][19];   a[ 2][20] = b[ 2][20];   a[ 2][21] = b[ 2][21];   a[ 2][22] = b[ 2][22];   a[ 2][23] = b[ 2][23];   a[ 2][24] = b[ 2][24];   a[ 2][25] = b[ 2][25];   a[ 2][26] = b[ 2][26];  
  a[ 3][ 0] = b[ 3][ 0];   a[ 3][ 1] = b[ 3][ 1];   a[ 3][ 2] = b[ 3][ 2];   a[ 3][ 3] = b[ 3][ 3];   a[ 3][ 4] = b[ 3][ 4];   a[ 3][ 5] = b[ 3][ 5];   a[ 3][ 6] = b[ 3][ 6];   a[ 3][ 7] = b[ 3][ 7];   a[ 3][ 8] = b[ 3][ 8];   a[ 3][ 9] = b[ 3][ 9];   a[ 3][10] = b[ 3][10];   a[ 3][11] = b[ 3][11];   a[ 3][12] = b[ 3][12];   a[ 3][13] = b[ 3][13];   a[ 3][14] = b[ 3][14];   a[ 3][15] = b[ 3][15];   a[ 3][16] = b[ 3][16];   a[ 3][17] = b[ 3][17];   a[ 3][18] = b[ 3][18];   a[ 3][19] = b[ 3][19];   a[ 3][20] = b[ 3][20];   a[ 3][21] = b[ 3][21];   a[ 3][22] = b[ 3][22];   a[ 3][23] = b[ 3][23];   a[ 3][24] = b[ 3][24];   a[ 3][25] = b[ 3][25];   a[ 3][26] = b[ 3][26];  
  a[ 4][ 0] = b[ 4][ 0];   a[ 4][ 1] = b[ 4][ 1];   a[ 4][ 2] = b[ 4][ 2];   a[ 4][ 3] = b[ 4][ 3];   a[ 4][ 4] = b[ 4][ 4];   a[ 4][ 5] = b[ 4][ 5];   a[ 4][ 6] = b[ 4][ 6];   a[ 4][ 7] = b[ 4][ 7];   a[ 4][ 8] = b[ 4][ 8];   a[ 4][ 9] = b[ 4][ 9];   a[ 4][10] = b[ 4][10];   a[ 4][11] = b[ 4][11];   a[ 4][12] = b[ 4][12];   a[ 4][13] = b[ 4][13];   a[ 4][14] = b[ 4][14];   a[ 4][15] = b[ 4][15];   a[ 4][16] = b[ 4][16];   a[ 4][17] = b[ 4][17];   a[ 4][18] = b[ 4][18];   a[ 4][19] = b[ 4][19];   a[ 4][20] = b[ 4][20];   a[ 4][21] = b[ 4][21];   a[ 4][22] = b[ 4][22];   a[ 4][23] = b[ 4][23];   a[ 4][24] = b[ 4][24];   a[ 4][25] = b[ 4][25];   a[ 4][26] = b[ 4][26];  
  a[ 5][ 0] = b[ 5][ 0];   a[ 5][ 1] = b[ 5][ 1];   a[ 5][ 2] = b[ 5][ 2];   a[ 5][ 3] = b[ 5][ 3];   a[ 5][ 4] = b[ 5][ 4];   a[ 5][ 5] = b[ 5][ 5];   a[ 5][ 6] = b[ 5][ 6];   a[ 5][ 7] = b[ 5][ 7];   a[ 5][ 8] = b[ 5][ 8];   a[ 5][ 9] = b[ 5][ 9];   a[ 5][10] = b[ 5][10];   a[ 5][11] = b[ 5][11];   a[ 5][12] = b[ 5][12];   a[ 5][13] = b[ 5][13];   a[ 5][14] = b[ 5][14];   a[ 5][15] = b[ 5][15];   a[ 5][16] = b[ 5][16];   a[ 5][17] = b[ 5][17];   a[ 5][18] = b[ 5][18];   a[ 5][19] = b[ 5][19];   a[ 5][20] = b[ 5][20];   a[ 5][21] = b[ 5][21];   a[ 5][22] = b[ 5][22];   a[ 5][23] = b[ 5][23];   a[ 5][24] = b[ 5][24];   a[ 5][25] = b[ 5][25];   a[ 5][26] = b[ 5][26];  
  a[ 6][ 0] = b[ 6][ 0];   a[ 6][ 1] = b[ 6][ 1];   a[ 6][ 2] = b[ 6][ 2];   a[ 6][ 3] = b[ 6][ 3];   a[ 6][ 4] = b[ 6][ 4];   a[ 6][ 5] = b[ 6][ 5];   a[ 6][ 6] = b[ 6][ 6];   a[ 6][ 7] = b[ 6][ 7];   a[ 6][ 8] = b[ 6][ 8];   a[ 6][ 9] = b[ 6][ 9];   a[ 6][10] = b[ 6][10];   a[ 6][11] = b[ 6][11];   a[ 6][12] = b[ 6][12];   a[ 6][13] = b[ 6][13];   a[ 6][14] = b[ 6][14];   a[ 6][15] = b[ 6][15];   a[ 6][16] = b[ 6][16];   a[ 6][17] = b[ 6][17];   a[ 6][18] = b[ 6][18];   a[ 6][19] = b[ 6][19];   a[ 6][20] = b[ 6][20];   a[ 6][21] = b[ 6][21];   a[ 6][22] = b[ 6][22];   a[ 6][23] = b[ 6][23];   a[ 6][24] = b[ 6][24];   a[ 6][25] = b[ 6][25];   a[ 6][26] = b[ 6][26];  
  a[ 7][ 0] = b[ 7][ 0];   a[ 7][ 1] = b[ 7][ 1];   a[ 7][ 2] = b[ 7][ 2];   a[ 7][ 3] = b[ 7][ 3];   a[ 7][ 4] = b[ 7][ 4];   a[ 7][ 5] = b[ 7][ 5];   a[ 7][ 6] = b[ 7][ 6];   a[ 7][ 7] = b[ 7][ 7];   a[ 7][ 8] = b[ 7][ 8];   a[ 7][ 9] = b[ 7][ 9];   a[ 7][10] = b[ 7][10];   a[ 7][11] = b[ 7][11];   a[ 7][12] = b[ 7][12];   a[ 7][13] = b[ 7][13];   a[ 7][14] = b[ 7][14];   a[ 7][15] = b[ 7][15];   a[ 7][16] = b[ 7][16];   a[ 7][17] = b[ 7][17];   a[ 7][18] = b[ 7][18];   a[ 7][19] = b[ 7][19];   a[ 7][20] = b[ 7][20];   a[ 7][21] = b[ 7][21];   a[ 7][22] = b[ 7][22];   a[ 7][23] = b[ 7][23];   a[ 7][24] = b[ 7][24];   a[ 7][25] = b[ 7][25];   a[ 7][26] = b[ 7][26];  
  a[ 8][ 0] = b[ 8][ 0];   a[ 8][ 1] = b[ 8][ 1];   a[ 8][ 2] = b[ 8][ 2];   a[ 8][ 3] = b[ 8][ 3];   a[ 8][ 4] = b[ 8][ 4];   a[ 8][ 5] = b[ 8][ 5];   a[ 8][ 6] = b[ 8][ 6];   a[ 8][ 7] = b[ 8][ 7];   a[ 8][ 8] = b[ 8][ 8];   a[ 8][ 9] = b[ 8][ 9];   a[ 8][10] = b[ 8][10];   a[ 8][11] = b[ 8][11];   a[ 8][12] = b[ 8][12];   a[ 8][13] = b[ 8][13];   a[ 8][14] = b[ 8][14];   a[ 8][15] = b[ 8][15];   a[ 8][16] = b[ 8][16];   a[ 8][17] = b[ 8][17];   a[ 8][18] = b[ 8][18];   a[ 8][19] = b[ 8][19];   a[ 8][20] = b[ 8][20];   a[ 8][21] = b[ 8][21];   a[ 8][22] = b[ 8][22];   a[ 8][23] = b[ 8][23];   a[ 8][24] = b[ 8][24];   a[ 8][25] = b[ 8][25];   a[ 8][26] = b[ 8][26];  
  a[ 9][ 0] = b[ 9][ 0];   a[ 9][ 1] = b[ 9][ 1];   a[ 9][ 2] = b[ 9][ 2];   a[ 9][ 3] = b[ 9][ 3];   a[ 9][ 4] = b[ 9][ 4];   a[ 9][ 5] = b[ 9][ 5];   a[ 9][ 6] = b[ 9][ 6];   a[ 9][ 7] = b[ 9][ 7];   a[ 9][ 8] = b[ 9][ 8];   a[ 9][ 9] = b[ 9][ 9];   a[ 9][10] = b[ 9][10];   a[ 9][11] = b[ 9][11];   a[ 9][12] = b[ 9][12];   a[ 9][13] = b[ 9][13];   a[ 9][14] = b[ 9][14];   a[ 9][15] = b[ 9][15];   a[ 9][16] = b[ 9][16];   a[ 9][17] = b[ 9][17];   a[ 9][18] = b[ 9][18];   a[ 9][19] = b[ 9][19];   a[ 9][20] = b[ 9][20];   a[ 9][21] = b[ 9][21];   a[ 9][22] = b[ 9][22];   a[ 9][23] = b[ 9][23];   a[ 9][24] = b[ 9][24];   a[ 9][25] = b[ 9][25];   a[ 9][26] = b[ 9][26];  
  a[10][ 0] = b[10][ 0];   a[10][ 1] = b[10][ 1];   a[10][ 2] = b[10][ 2];   a[10][ 3] = b[10][ 3];   a[10][ 4] = b[10][ 4];   a[10][ 5] = b[10][ 5];   a[10][ 6] = b[10][ 6];   a[10][ 7] = b[10][ 7];   a[10][ 8] = b[10][ 8];   a[10][ 9] = b[10][ 9];   a[10][10] = b[10][10];   a[10][11] = b[10][11];   a[10][12] = b[10][12];   a[10][13] = b[10][13];   a[10][14] = b[10][14];   a[10][15] = b[10][15];   a[10][16] = b[10][16];   a[10][17] = b[10][17];   a[10][18] = b[10][18];   a[10][19] = b[10][19];   a[10][20] = b[10][20];   a[10][21] = b[10][21];   a[10][22] = b[10][22];   a[10][23] = b[10][23];   a[10][24] = b[10][24];   a[10][25] = b[10][25];   a[10][26] = b[10][26];  
  a[11][ 0] = b[11][ 0];   a[11][ 1] = b[11][ 1];   a[11][ 2] = b[11][ 2];   a[11][ 3] = b[11][ 3];   a[11][ 4] = b[11][ 4];   a[11][ 5] = b[11][ 5];   a[11][ 6] = b[11][ 6];   a[11][ 7] = b[11][ 7];   a[11][ 8] = b[11][ 8];   a[11][ 9] = b[11][ 9];   a[11][10] = b[11][10];   a[11][11] = b[11][11];   a[11][12] = b[11][12];   a[11][13] = b[11][13];   a[11][14] = b[11][14];   a[11][15] = b[11][15];   a[11][16] = b[11][16];   a[11][17] = b[11][17];   a[11][18] = b[11][18];   a[11][19] = b[11][19];   a[11][20] = b[11][20];   a[11][21] = b[11][21];   a[11][22] = b[11][22];   a[11][23] = b[11][23];   a[11][24] = b[11][24];   a[11][25] = b[11][25];   a[11][26] = b[11][26];  
  a[12][ 0] = b[12][ 0];   a[12][ 1] = b[12][ 1];   a[12][ 2] = b[12][ 2];   a[12][ 3] = b[12][ 3];   a[12][ 4] = b[12][ 4];   a[12][ 5] = b[12][ 5];   a[12][ 6] = b[12][ 6];   a[12][ 7] = b[12][ 7];   a[12][ 8] = b[12][ 8];   a[12][ 9] = b[12][ 9];   a[12][10] = b[12][10];   a[12][11] = b[12][11];   a[12][12] = b[12][12];   a[12][13] = b[12][13];   a[12][14] = b[12][14];   a[12][15] = b[12][15];   a[12][16] = b[12][16];   a[12][17] = b[12][17];   a[12][18] = b[12][18];   a[12][19] = b[12][19];   a[12][20] = b[12][20];   a[12][21] = b[12][21];   a[12][22] = b[12][22];   a[12][23] = b[12][23];   a[12][24] = b[12][24];   a[12][25] = b[12][25];   a[12][26] = b[12][26];  
  a[13][ 0] = b[13][ 0];   a[13][ 1] = b[13][ 1];   a[13][ 2] = b[13][ 2];   a[13][ 3] = b[13][ 3];   a[13][ 4] = b[13][ 4];   a[13][ 5] = b[13][ 5];   a[13][ 6] = b[13][ 6];   a[13][ 7] = b[13][ 7];   a[13][ 8] = b[13][ 8];   a[13][ 9] = b[13][ 9];   a[13][10] = b[13][10];   a[13][11] = b[13][11];   a[13][12] = b[13][12];   a[13][13] = b[13][13];   a[13][14] = b[13][14];   a[13][15] = b[13][15];   a[13][16] = b[13][16];   a[13][17] = b[13][17];   a[13][18] = b[13][18];   a[13][19] = b[13][19];   a[13][20] = b[13][20];   a[13][21] = b[13][21];   a[13][22] = b[13][22];   a[13][23] = b[13][23];   a[13][24] = b[13][24];   a[13][25] = b[13][25];   a[13][26] = b[13][26];  
  a[14][ 0] = b[14][ 0];   a[14][ 1] = b[14][ 1];   a[14][ 2] = b[14][ 2];   a[14][ 3] = b[14][ 3];   a[14][ 4] = b[14][ 4];   a[14][ 5] = b[14][ 5];   a[14][ 6] = b[14][ 6];   a[14][ 7] = b[14][ 7];   a[14][ 8] = b[14][ 8];   a[14][ 9] = b[14][ 9];   a[14][10] = b[14][10];   a[14][11] = b[14][11];   a[14][12] = b[14][12];   a[14][13] = b[14][13];   a[14][14] = b[14][14];   a[14][15] = b[14][15];   a[14][16] = b[14][16];   a[14][17] = b[14][17];   a[14][18] = b[14][18];   a[14][19] = b[14][19];   a[14][20] = b[14][20];   a[14][21] = b[14][21];   a[14][22] = b[14][22];   a[14][23] = b[14][23];   a[14][24] = b[14][24];   a[14][25] = b[14][25];   a[14][26] = b[14][26];  
  a[15][ 0] = b[15][ 0];   a[15][ 1] = b[15][ 1];   a[15][ 2] = b[15][ 2];   a[15][ 3] = b[15][ 3];   a[15][ 4] = b[15][ 4];   a[15][ 5] = b[15][ 5];   a[15][ 6] = b[15][ 6];   a[15][ 7] = b[15][ 7];   a[15][ 8] = b[15][ 8];   a[15][ 9] = b[15][ 9];   a[15][10] = b[15][10];   a[15][11] = b[15][11];   a[15][12] = b[15][12];   a[15][13] = b[15][13];   a[15][14] = b[15][14];   a[15][15] = b[15][15];   a[15][16] = b[15][16];   a[15][17] = b[15][17];   a[15][18] = b[15][18];   a[15][19] = b[15][19];   a[15][20] = b[15][20];   a[15][21] = b[15][21];   a[15][22] = b[15][22];   a[15][23] = b[15][23];   a[15][24] = b[15][24];   a[15][25] = b[15][25];   a[15][26] = b[15][26];  
  a[16][ 0] = b[16][ 0];   a[16][ 1] = b[16][ 1];   a[16][ 2] = b[16][ 2];   a[16][ 3] = b[16][ 3];   a[16][ 4] = b[16][ 4];   a[16][ 5] = b[16][ 5];   a[16][ 6] = b[16][ 6];   a[16][ 7] = b[16][ 7];   a[16][ 8] = b[16][ 8];   a[16][ 9] = b[16][ 9];   a[16][10] = b[16][10];   a[16][11] = b[16][11];   a[16][12] = b[16][12];   a[16][13] = b[16][13];   a[16][14] = b[16][14];   a[16][15] = b[16][15];   a[16][16] = b[16][16];   a[16][17] = b[16][17];   a[16][18] = b[16][18];   a[16][19] = b[16][19];   a[16][20] = b[16][20];   a[16][21] = b[16][21];   a[16][22] = b[16][22];   a[16][23] = b[16][23];   a[16][24] = b[16][24];   a[16][25] = b[16][25];   a[16][26] = b[16][26];  
  a[17][ 0] = b[17][ 0];   a[17][ 1] = b[17][ 1];   a[17][ 2] = b[17][ 2];   a[17][ 3] = b[17][ 3];   a[17][ 4] = b[17][ 4];   a[17][ 5] = b[17][ 5];   a[17][ 6] = b[17][ 6];   a[17][ 7] = b[17][ 7];   a[17][ 8] = b[17][ 8];   a[17][ 9] = b[17][ 9];   a[17][10] = b[17][10];   a[17][11] = b[17][11];   a[17][12] = b[17][12];   a[17][13] = b[17][13];   a[17][14] = b[17][14];   a[17][15] = b[17][15];   a[17][16] = b[17][16];   a[17][17] = b[17][17];   a[17][18] = b[17][18];   a[17][19] = b[17][19];   a[17][20] = b[17][20];   a[17][21] = b[17][21];   a[17][22] = b[17][22];   a[17][23] = b[17][23];   a[17][24] = b[17][24];   a[17][25] = b[17][25];   a[17][26] = b[17][26];  
  a[18][ 0] = b[18][ 0];   a[18][ 1] = b[18][ 1];   a[18][ 2] = b[18][ 2];   a[18][ 3] = b[18][ 3];   a[18][ 4] = b[18][ 4];   a[18][ 5] = b[18][ 5];   a[18][ 6] = b[18][ 6];   a[18][ 7] = b[18][ 7];   a[18][ 8] = b[18][ 8];   a[18][ 9] = b[18][ 9];   a[18][10] = b[18][10];   a[18][11] = b[18][11];   a[18][12] = b[18][12];   a[18][13] = b[18][13];   a[18][14] = b[18][14];   a[18][15] = b[18][15];   a[18][16] = b[18][16];   a[18][17] = b[18][17];   a[18][18] = b[18][18];   a[18][19] = b[18][19];   a[18][20] = b[18][20];   a[18][21] = b[18][21];   a[18][22] = b[18][22];   a[18][23] = b[18][23];   a[18][24] = b[18][24];   a[18][25] = b[18][25];   a[18][26] = b[18][26];  
  a[19][ 0] = b[19][ 0];   a[19][ 1] = b[19][ 1];   a[19][ 2] = b[19][ 2];   a[19][ 3] = b[19][ 3];   a[19][ 4] = b[19][ 4];   a[19][ 5] = b[19][ 5];   a[19][ 6] = b[19][ 6];   a[19][ 7] = b[19][ 7];   a[19][ 8] = b[19][ 8];   a[19][ 9] = b[19][ 9];   a[19][10] = b[19][10];   a[19][11] = b[19][11];   a[19][12] = b[19][12];   a[19][13] = b[19][13];   a[19][14] = b[19][14];   a[19][15] = b[19][15];   a[19][16] = b[19][16];   a[19][17] = b[19][17];   a[19][18] = b[19][18];   a[19][19] = b[19][19];   a[19][20] = b[19][20];   a[19][21] = b[19][21];   a[19][22] = b[19][22];   a[19][23] = b[19][23];   a[19][24] = b[19][24];   a[19][25] = b[19][25];   a[19][26] = b[19][26];  
  a[20][ 0] = b[20][ 0];   a[20][ 1] = b[20][ 1];   a[20][ 2] = b[20][ 2];   a[20][ 3] = b[20][ 3];   a[20][ 4] = b[20][ 4];   a[20][ 5] = b[20][ 5];   a[20][ 6] = b[20][ 6];   a[20][ 7] = b[20][ 7];   a[20][ 8] = b[20][ 8];   a[20][ 9] = b[20][ 9];   a[20][10] = b[20][10];   a[20][11] = b[20][11];   a[20][12] = b[20][12];   a[20][13] = b[20][13];   a[20][14] = b[20][14];   a[20][15] = b[20][15];   a[20][16] = b[20][16];   a[20][17] = b[20][17];   a[20][18] = b[20][18];   a[20][19] = b[20][19];   a[20][20] = b[20][20];   a[20][21] = b[20][21];   a[20][22] = b[20][22];   a[20][23] = b[20][23];   a[20][24] = b[20][24];   a[20][25] = b[20][25];   a[20][26] = b[20][26];  
  a[21][ 0] = b[21][ 0];   a[21][ 1] = b[21][ 1];   a[21][ 2] = b[21][ 2];   a[21][ 3] = b[21][ 3];   a[21][ 4] = b[21][ 4];   a[21][ 5] = b[21][ 5];   a[21][ 6] = b[21][ 6];   a[21][ 7] = b[21][ 7];   a[21][ 8] = b[21][ 8];   a[21][ 9] = b[21][ 9];   a[21][10] = b[21][10];   a[21][11] = b[21][11];   a[21][12] = b[21][12];   a[21][13] = b[21][13];   a[21][14] = b[21][14];   a[21][15] = b[21][15];   a[21][16] = b[21][16];   a[21][17] = b[21][17];   a[21][18] = b[21][18];   a[21][19] = b[21][19];   a[21][20] = b[21][20];   a[21][21] = b[21][21];   a[21][22] = b[21][22];   a[21][23] = b[21][23];   a[21][24] = b[21][24];   a[21][25] = b[21][25];   a[21][26] = b[21][26];  
  a[22][ 0] = b[22][ 0];   a[22][ 1] = b[22][ 1];   a[22][ 2] = b[22][ 2];   a[22][ 3] = b[22][ 3];   a[22][ 4] = b[22][ 4];   a[22][ 5] = b[22][ 5];   a[22][ 6] = b[22][ 6];   a[22][ 7] = b[22][ 7];   a[22][ 8] = b[22][ 8];   a[22][ 9] = b[22][ 9];   a[22][10] = b[22][10];   a[22][11] = b[22][11];   a[22][12] = b[22][12];   a[22][13] = b[22][13];   a[22][14] = b[22][14];   a[22][15] = b[22][15];   a[22][16] = b[22][16];   a[22][17] = b[22][17];   a[22][18] = b[22][18];   a[22][19] = b[22][19];   a[22][20] = b[22][20];   a[22][21] = b[22][21];   a[22][22] = b[22][22];   a[22][23] = b[22][23];   a[22][24] = b[22][24];   a[22][25] = b[22][25];   a[22][26] = b[22][26];  
  a[23][ 0] = b[23][ 0];   a[23][ 1] = b[23][ 1];   a[23][ 2] = b[23][ 2];   a[23][ 3] = b[23][ 3];   a[23][ 4] = b[23][ 4];   a[23][ 5] = b[23][ 5];   a[23][ 6] = b[23][ 6];   a[23][ 7] = b[23][ 7];   a[23][ 8] = b[23][ 8];   a[23][ 9] = b[23][ 9];   a[23][10] = b[23][10];   a[23][11] = b[23][11];   a[23][12] = b[23][12];   a[23][13] = b[23][13];   a[23][14] = b[23][14];   a[23][15] = b[23][15];   a[23][16] = b[23][16];   a[23][17] = b[23][17];   a[23][18] = b[23][18];   a[23][19] = b[23][19];   a[23][20] = b[23][20];   a[23][21] = b[23][21];   a[23][22] = b[23][22];   a[23][23] = b[23][23];   a[23][24] = b[23][24];   a[23][25] = b[23][25];   a[23][26] = b[23][26];  
  a[24][ 0] = b[24][ 0];   a[24][ 1] = b[24][ 1];   a[24][ 2] = b[24][ 2];   a[24][ 3] = b[24][ 3];   a[24][ 4] = b[24][ 4];   a[24][ 5] = b[24][ 5];   a[24][ 6] = b[24][ 6];   a[24][ 7] = b[24][ 7];   a[24][ 8] = b[24][ 8];   a[24][ 9] = b[24][ 9];   a[24][10] = b[24][10];   a[24][11] = b[24][11];   a[24][12] = b[24][12];   a[24][13] = b[24][13];   a[24][14] = b[24][14];   a[24][15] = b[24][15];   a[24][16] = b[24][16];   a[24][17] = b[24][17];   a[24][18] = b[24][18];   a[24][19] = b[24][19];   a[24][20] = b[24][20];   a[24][21] = b[24][21];   a[24][22] = b[24][22];   a[24][23] = b[24][23];   a[24][24] = b[24][24];   a[24][25] = b[24][25];   a[24][26] = b[24][26];  
  a[25][ 0] = b[25][ 0];   a[25][ 1] = b[25][ 1];   a[25][ 2] = b[25][ 2];   a[25][ 3] = b[25][ 3];   a[25][ 4] = b[25][ 4];   a[25][ 5] = b[25][ 5];   a[25][ 6] = b[25][ 6];   a[25][ 7] = b[25][ 7];   a[25][ 8] = b[25][ 8];   a[25][ 9] = b[25][ 9];   a[25][10] = b[25][10];   a[25][11] = b[25][11];   a[25][12] = b[25][12];   a[25][13] = b[25][13];   a[25][14] = b[25][14];   a[25][15] = b[25][15];   a[25][16] = b[25][16];   a[25][17] = b[25][17];   a[25][18] = b[25][18];   a[25][19] = b[25][19];   a[25][20] = b[25][20];   a[25][21] = b[25][21];   a[25][22] = b[25][22];   a[25][23] = b[25][23];   a[25][24] = b[25][24];   a[25][25] = b[25][25];   a[25][26] = b[25][26];  
  a[26][ 0] = b[26][ 0];   a[26][ 1] = b[26][ 1];   a[26][ 2] = b[26][ 2];   a[26][ 3] = b[26][ 3];   a[26][ 4] = b[26][ 4];   a[26][ 5] = b[26][ 5];   a[26][ 6] = b[26][ 6];   a[26][ 7] = b[26][ 7];   a[26][ 8] = b[26][ 8];   a[26][ 9] = b[26][ 9];   a[26][10] = b[26][10];   a[26][11] = b[26][11];   a[26][12] = b[26][12];   a[26][13] = b[26][13];   a[26][14] = b[26][14];   a[26][15] = b[26][15];   a[26][16] = b[26][16];   a[26][17] = b[26][17];   a[26][18] = b[26][18];   a[26][19] = b[26][19];   a[26][20] = b[26][20];   a[26][21] = b[26][21];   a[26][22] = b[26][22];   a[26][23] = b[26][23];   a[26][24] = b[26][24];   a[26][25] = b[26][25];   a[26][26] = b[26][26];
}
void mat_multiply_matrix_d27(double m[27][27],const double a[27][27],const double b[27][27])
{
  int i, j;                   /* a_ij */
  double c[27][27];
  for (i = 0; i < 27; i++) {
    for (j = 0; j < 27; j++) {
      c[i][j] =
        a[i][ 0] * b[ 0][j] +  a[i][ 1] * b[ 1][j] +  a[i][ 2] * b[ 2][j] +  a[i][ 3] * b[ 3][j] +  a[i][ 4] * b[ 4][j] +  a[i][ 5] * b[ 5][j] +  a[i][ 6] * b[ 6][j] +  a[i][ 7] * b[ 7][j] +  a[i][ 8] * b[ 8][j] +  a[i][ 9] * b[ 9][j] +  a[i][10] * b[10][j] +  a[i][11] * b[11][j] +  a[i][12] * b[12][j] +  a[i][13] * b[13][j] +  a[i][14] * b[14][j] +  a[i][15] * b[15][j] +  a[i][16] * b[16][j] +  a[i][17] * b[17][j] +  a[i][18] * b[18][j] +  a[i][19] * b[19][j] +  a[i][20] * b[20][j] +  a[i][21] * b[21][j] +  a[i][22] * b[22][j] +  a[i][23] * b[23][j] +  a[i][24] * b[24][j] +  a[i][25] * b[25][j] +  a[i][26] * b[26][j];
    }
  }
  mat_copy_mat_d27(m, c);
}

int mat_check_zero_vector_d27(const double a[27], const double prec)
{
  int i, all_zero = 1;
  for (i=0; i<27; i++)
  {
    if (mat_Dabs(a[i]) > prec)
    {
      all_zero = 0;
      break;
    }
  }
  return all_zero;
}

int mat_all_close_vector_d27(const double a[27], const double b[27], const double prec)
{
  int i, all_close = 1;
  for (i=0; i<27; i++)
  {
    if (mat_Dabs(a[i] - b[i]) > prec)
    {
      all_close = 0;
      break;
    }
  }
  return all_close;
}

void mat_copy_vector_d27(double a[27], const double b[27])
{
  int i;
  for (i=0; i<27; i++)
  {
    a[i] = b[i];
  }
}

/* m=axb */
void mat_multiply_matrix_d3(double m[3][3],
          const double a[3][3],
          const double b[3][3])
{
  int i, j;                   /* a_ij */
  double c[3][3];
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      c[i][j] =
  a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  mat_copy_matrix_d3(m, c);
}

void mat_multiply_matrix_i3(int m[3][3],
          const int a[3][3],
          const int b[3][3])
{
  int i, j;                   /* a_ij */
  int c[3][3];
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      c[i][j] =
  a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  mat_copy_matrix_i3(m, c);
}

void mat_multiply_matrix_di3(double m[3][3],
           const double a[3][3],
           const int b[3][3])
{
  int i, j;                   /* a_ij */
  double c[3][3];
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      c[i][j] =
  a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  mat_copy_matrix_d3(m, c);
}

void mat_multiply_matrix_id3( double m[3][3],
            const int a[3][3],
            const double b[3][3])
{
  int i, j;                   /* a_ij */
  double c[3][3];
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      c[i][j] =
  a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  mat_copy_matrix_d3(m, c);
}

void mat_multiply_matrix_vector_i3(int v[3],
           const int a[3][3],
           const int b[3])
{
  int i;
  int c[3];
  for (i = 0; i < 3; i++)
    c[i] = a[i][0] * b[0] + a[i][1] * b[1] + a[i][2] * b[2];
  for (i = 0; i < 3; i++)
    v[i] = c[i];
}

void mat_multiply_matrix_vector_d3(double v[3],
           const double a[3][3],
           const double b[3])
{
  int i;
  double c[3];
  for (i = 0; i < 3; i++)
    c[i] = a[i][0] * b[0] + a[i][1] * b[1] + a[i][2] * b[2];
  for (i = 0; i < 3; i++)
    v[i] = c[i];
}

void mat_multiply_matrix_vector_id3(double v[3],
            const int a[3][3],
            const double b[3])
{
  int i;
  double c[3];
  for (i = 0; i < 3; i++)
    c[i] = a[i][0] * b[0] + a[i][1] * b[1] + a[i][2] * b[2];
  for (i = 0; i < 3; i++)
    v[i] = c[i];
}

void mat_multiply_matrix_vector_di3(double v[3],
            const double a[3][3],
            const int b[3])
{
  int i;
  double c[3];
  for (i = 0; i < 3; i++)
    c[i] = a[i][0] * b[0] + a[i][1] * b[1] + a[i][2] * b[2];
  for (i = 0; i < 3; i++)
    v[i] = c[i];
}

void mat_add_matrix_i3( int m[3][3],
      const int a[3][3],
      const int b[3][3] )
{
  int i, j;
  for ( i = 0; i < 3; i++ ) {
    for ( j = 0; j < 3; j++ ) {
      m[i][j] = a[i][j] + b[i][j];
    }
  }
}


void mat_cast_matrix_3i_to_3d(double m[3][3], const int a[3][3])
{
  m[0][0] = a[0][0];
  m[0][1] = a[0][1];
  m[0][2] = a[0][2];
  m[1][0] = a[1][0];
  m[1][1] = a[1][1];
  m[1][2] = a[1][2];
  m[2][0] = a[2][0];
  m[2][1] = a[2][1];
  m[2][2] = a[2][2];
}

void mat_cast_matrix_3d_to_3i(int m[3][3], const double a[3][3])
{
  m[0][0] = mat_Nint(a[0][0]);
  m[0][1] = mat_Nint(a[0][1]);
  m[0][2] = mat_Nint(a[0][2]);
  m[1][0] = mat_Nint(a[1][0]);
  m[1][1] = mat_Nint(a[1][1]);
  m[1][2] = mat_Nint(a[1][2]);
  m[2][0] = mat_Nint(a[2][0]);
  m[2][1] = mat_Nint(a[2][1]);
  m[2][2] = mat_Nint(a[2][2]);
}

/* m^-1 */
/* ruby code for auto generating */
/* 3.times {|i| 3.times {|j| */
/*       puts "m[#{j}][#{i}]=(a[#{(i+1)%3}][#{(j+1)%3}]*a[#{(i+2)%3}][#{(j+2)%3}] */
/*   -a[#{(i+1)%3}][#{(j+2)%3}]*a[#{(i+2)%3}][#{(j+1)%3}])/det;" */
/* }} */
int mat_inverse_matrix_d3(double m[3][3],
        const double a[3][3],
        const double precision)
{
  double det;
  double c[3][3];
  det = mat_get_determinant_d3(a);
  if (mat_Dabs(det) < precision) {
    warning_print("spglib: No inverse matrix (det=%f)\n", det);
    warning_print("No inverse matrix\n");
    return 0;
  }

  c[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
  c[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
  c[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
  c[0][1] = (a[2][1] * a[0][2] - a[2][2] * a[0][1]) / det;
  c[1][1] = (a[2][2] * a[0][0] - a[2][0] * a[0][2]) / det;
  c[2][1] = (a[2][0] * a[0][1] - a[2][1] * a[0][0]) / det;
  c[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
  c[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
  c[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
  mat_copy_matrix_d3(m, c);
  return 1;
}

/* m = b a b ^-1 */
int mat_get_similar_matrix_d3(double m[3][3],
            const double a[3][3],
            const double b[3][3],
            const double precision)
{
  double c[3][3];
  if (!mat_inverse_matrix_d3(c, b, precision)) {
    warning_print("spglib: No similar matrix due to 0 determinant.\n");
    warning_print("No similar matrix due to 0 determinant.\n");
    return 0;
  }
  mat_multiply_matrix_d3(m, a, c);
  mat_multiply_matrix_d3(m, b, m);
  return 1;
}

void mat_transpose_matrix_d3(double a[3][3], const double b[3][3])
{
  double c[3][3];
  c[0][0] = b[0][0];
  c[0][1] = b[1][0];
  c[0][2] = b[2][0];
  c[1][0] = b[0][1];
  c[1][1] = b[1][1];
  c[1][2] = b[2][1];
  c[2][0] = b[0][2];
  c[2][1] = b[1][2];
  c[2][2] = b[2][2];
  mat_copy_matrix_d3(a, c);
}

void mat_transpose_matrix_i3(int a[3][3], const int b[3][3])
{
  int c[3][3];
  c[0][0] = b[0][0];
  c[0][1] = b[1][0];
  c[0][2] = b[2][0];
  c[1][0] = b[0][1];
  c[1][1] = b[1][1];
  c[1][2] = b[2][1];
  c[2][0] = b[0][2];
  c[2][1] = b[1][2];
  c[2][2] = b[2][2];
  mat_copy_matrix_i3(a, c);
}

void mat_get_metric( double metric[3][3],
         const double lattice[3][3])
{
  double lattice_t[3][3];
  mat_transpose_matrix_d3(lattice_t, lattice);
  mat_multiply_matrix_d3(metric, lattice_t, lattice);
}

double mat_norm_squared_d3( const double a[3] )
{
  return a[0]*a[0]+a[1]*a[1]+a[2]*a[2];
}

int mat_norm_squared_i3( const int a[3] )
{
  return a[0]*a[0]+a[1]*a[1]+a[2]*a[2];
}

double mat_Dabs(const double a)
{
  if (a < 0.0)
    return -a;
  else
    return a;
}

void mat_normalize_by_abs_vector_d27(double a[27], double precesion)
{
  int i;
  double value, max = 0;
  for (i=0; i<27; i++)
    if (mat_Dabs(a[i]) > max)
      {
        max = mat_Dabs(a[i]);
        value = a[i];
      }
  if (max > precesion)
    for (i = 0; i < 27; i++)
       a[i] /= value;
}

int mat_Nint(const double a)
{
  if (a < 0.0)
    return (int) (a - 0.5);
  else
    return (int) (a + 0.5);
}

double mat_Dmod1(const double a)
{
  if (a < 0.0)
    return a + 1.0 - (int) a;
  else
    return a - (int) a;
}

int mat_is_int_matrix( const double mat[3][3], double symprec )
{
  int i,j ;
  for ( i = 0; i < 3; i++ ) {
    for ( j = 0; j < 3; j++ ) {
      if ( mat_Dabs( mat_Nint( mat[i][j] ) - mat[i][j] ) > symprec ) {
  return 0;
      }
    }
  }
  return 1;
}

int get_index_from_array(const int *a, const int length, const int value)
{
  int i, index=-1;
  for (i=0; i<length; i++)
  {
    if (a[i] == value)
    {
      index = i;
      break;
    }
  }
  return index;
}

int get_index_from_vectors_i3(const int (*a)[3], const int length, const int value[3])
{
  int i, index=-1;
  for (i=0; i<length; i++)
  {
    if (a[i][0] == value[0] &&
        a[i][1] == value[1] &&
        a[i][2] == value[2])
    {
      index = i;
      break;
    }
  }
  return index;
}

int get_index_from_rots(const int (*a)[3][3], const int length, const int value[3][3])
{
  int i, index=-1;
  for (i=0; i<length; i++)
  {
    if (a[i][0][0] == value[0][0] && a[i][0][1] == value[0][1] && a[i][0][2] == value[0][2] &&
        a[i][1][0] == value[1][0] && a[i][1][1] == value[1][1] && a[i][1][2] == value[1][2] &&
        a[i][2][0] == value[2][0] && a[i][2][1] == value[2][1] && a[i][2][2] == value[2][2])
    {
      index = i;
      break;
    }
  }
  return index;
}

int get_index_from_pos_vector_d3(const double *a[3], const int length, const double value[3], const double precesion)
{
  int i, index=-1;
  for (i=0; i<length; i++)
  {
    if (mat_Dabs(a[i][0] - mat_Nint(value[0])) < precesion &&
        mat_Dabs(a[i][1] - mat_Nint(value[1])) < precesion &&
        mat_Dabs(a[i][2] - mat_Nint(value[2])) < precesion)
    {
      index = i;
      break;
    }
  }
  return index;
}

int array_unique(int *arr, const int *array_orig, const int length)
{
  int i, size=0;
  init_ivector(arr, length, -1); // Init with -1
  for (i = 0; i < length; i++){
    if (get_index_from_array(arr, size, array_orig[i]) == -1)
    { //not exist
      arr[size++] = array_orig[i];
    }
  }
  return size;
}

// int  gaussian(double **b, 
//         int *IndexIndependent,
//         double **a, 
//         const int row, 
//         const int column,
//         const double prec)
// {
//   int Ndependent, Nindependent;
//   int  i,j,k,irow;
//   int *Indexdependent = ivector(column);
//   double *swap_ik = dvector(column);
//   
//   Nindependent=0;
//   Ndependent=0;
//   for (i=0; i<column; i++)
//   {
//     swap_ik[i]=0.0;
//   IndexIndependent[i] = 0;
//   }
//   irow=0;
//   for(k=0; k<column; k++){
//     // swap row i and row k
//     for (i=irow+1; i<row; i++)
//     {
//       if (mat_Dabs(a[i][k]) - mat_Dabs(a[irow][k]) > prec)
//       {
//         for (j=k; j<column; j++)
//         {  
//           swap_ik[j]=a[irow][j];
//           a[irow][j]=a[i][j];
//           a[i][j]=swap_ik[j];
//         }
//       }
//     }
//     if(mat_Dabs(a[irow][k])>prec)
//     {    
//       Indexdependent[Ndependent++]=k;
//       for (j=column-1; j>k; j--)
//         a[irow][j] /= a[irow][k];
//     a[irow][k] = 1.0;
//       for (i=0; i<irow; i++)
//       {
//         for (j=column-1; j>k; j--)
//           a[i][j] -= a[irow][j] / a[irow][k] * a[i][k];
//         a[i][k]=0.0;
//       }
//       for(i=irow+1; i<row; i++)
//       {
//         for(j=column-1; j>k; j--)
//            a[i][j] -= a[irow][j] / a[irow][k] * a[i][k];
//         a[i][k] = 0.0;
//       }
//     irow++;
//     }
//     else
//     {
//       IndexIndependent[Nindependent++]=k;
//     } 
//   }
//   for (i=0; i<column; i++)
//     for (j=0; j<column; j++)
//       b[i][j] = 0.0;
//   if(Nindependent > 0)
//   {
//     for (i=0; i<Ndependent; i++)
//     {
//       for (j=0; j < Nindependent; j++)
//         b[Indexdependent[i]][j]=-a[i][IndexIndependent[j]];
//     }
//     for (j=0; j<Nindependent; j++)
//     {
//       b[IndexIndependent[j]][j]=1.0;
//     }
//   }
//   free_ivector(Indexdependent);
//   free_dvector(swap_ik);
//   return Nindependent;
// }





int  gaussian(double **b, 
        int *IndexIndependent,
        double **a, 
        const int row, 
        const int column,
        const double prec) //with row and column pivoting
{
  int Ndependent, Nindependent;
  int  i,j,k,irow;
  int *Indexdependent = ivector(column);
  double *swap_ik = dvector(column);
  
  Nindependent=0;
  Ndependent=0;
  for (i=0; i<column; i++)
  {
    swap_ik[i]=0.0;
  IndexIndependent[i] = 0;
  }
  irow=0;
  for(k=0; k<column; k++){
    // swap row i and row k
    for (i=irow+1; i<row; i++)
    {
      if (mat_Dabs(a[i][k]) - mat_Dabs(a[irow][k]) > prec)
      {
        for (j=k; j<column; j++)
        {  
          swap_ik[j]=a[irow][j];
          a[irow][j]=a[i][j];
          a[i][j]=swap_ik[j];
        }
      }
    }
    if(mat_Dabs(a[irow][k]) > prec * 100)
    {    
      Indexdependent[Ndependent++]=k;
      for (j=column-1; j>k; j--)
        a[irow][j] /= a[irow][k];
    a[irow][k] = 1.0;
      for (i=0; i<irow; i++)
      {
        for (j=column-1; j>k; j--)
          a[i][j] -= a[irow][j] / a[irow][k] * a[i][k];
        a[i][k]=0.0;
      }
      for(i=irow+1; i<row; i++)
      {
        for(j=column-1; j>k; j--)
           a[i][j] -= a[irow][j] / a[irow][k] * a[i][k];
        a[i][k] = 0.0;
      }
    irow++;
    }
    else
    {
      IndexIndependent[Nindependent++]=k;
    } 
  }
  for (i=0; i<column; i++)
    for (j=0; j<column; j++)
      b[i][j] = 0.0;
  if(Nindependent > 0)
  {
    for (i=0; i<Ndependent; i++)
    {
      for (j=0; j < Nindependent; j++)
        b[Indexdependent[i]][j]=-a[i][IndexIndependent[j]];
    }
    for (j=0; j<Nindependent; j++)
    {
      b[IndexIndependent[j]][j]=1.0;
    }
  }
  free_ivector(Indexdependent);
  free_dvector(swap_ik);
  return Nindependent;
}