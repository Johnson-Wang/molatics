/* mathfunc.h */
/* Copyright (C) 2015 Wang Xinjiang */

#ifndef __mathfunc_H__
#define __mathfunc_H__

#define FREE_ARG char*

typedef struct {
  int size;
  int (*mat)[3][3];
} MatINT;

typedef struct {
  int n;
  int *vec;
} VecArbiLenINT;

typedef struct {
  int n;
  double *vec;
} VecArbiLenDBL;

typedef struct {
  int row;
  int column;
  int **mat;
} MatArbiLenINT;

typedef struct {
  int row;
  int column;
  double **mat;
} MatArbiLenDBL;

typedef struct {
  int row;
  int column;
  int depth;
  int ***f3;
} F3ArbiLenINT;

typedef struct {
  int row;
  int column;
  int depth;
  double ***f3;
} F3ArbiLenDBL;

typedef struct {
  int size;
  double (*vec)[3];
} VecDBL;

typedef struct {
  int size;
  int (*tri)[3];
} Triplet;

typedef struct {
  int size;
  int (*rot)[3][3];
  double (*trans)[3];
} Symmetry;

typedef struct {
  int (*rot)[3][3];
  int size;
} PointSymmetry;

typedef struct {
  int (*rot)[27][27];
  int size;
} MatTensorINT;

typedef struct {
  double (*rot)[27][27];
  int size;
} MatTensorDBL;

Symmetry * sym_alloc_symmetry( const int size );
void sym_free_symmetry( Symmetry * symmetry );
PointSymmetry *sym_alloc_point_symmetry(const int size);
void sym_free_point_symmetry(PointSymmetry * ps);
MatTensorINT * alloc_MatTensorINT( const int size );
void free_MatTensorINT( MatTensorINT * matt );
MatTensorDBL * alloc_MatTensorDBL( const int size );
void free_MatTensorDBL( MatTensorDBL * matt );

void runerror(char error_text[]);
int *ivector(long n);
/* allocate an int vector with subscript range v[0..n] */
double *dvector(long n);
/* allocate a double vector with subscript range v[nl..nh] */
int **imatrix(long nr, long nc);
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
double **dmatrix(long nr, long nc);
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
int ***if3tensor(long nr, long nc, long nd);
/* allocate a int 3tensor with range t[0..nr][0..nc][0..nd] */
double ***df3tensor(long nr, long nc, long nd);
/* allocate a double 3tensor with range t[0..nr][0..nc][0..nd] */

void init_ivector(int *v, const long length, int value);
/* init an int vector allocated with ivector() */
void init_dvector(double *v, const long length, double value);
/* init a double vector allocated with dvector() */
void init_dmatrix(double **m, const long row, const long column, double value);
/* init a double matrix allocated by dmatrix() */
void init_imatrix(int **m, const long row, const long column, int value);
/* init an int matrix allocated by imatrix() */
void init_if3tensor(int ***t, const long row, const long column, const long depth, int value);
/* init a int f3tensor allocated by f3tensor() */
void init_df3tensor(double ***t, const long row, const long column, const long depth, double value);
/* init a double f3tensor allocated by f3tensor() */

void free_ivector(int *v);
/* free an int vector allocated with ivector() */
void free_dvector(double *v);
/* free a double vector allocated with dvector() */
void free_imatrix(int **m);
/* free an int matrix allocated by imatrix() */
void free_dmatrix(double **m);
/* free a double matrix allocated by dmatrix() */
void free_if3tensor(int ***t);
/* free a double f3tensor allocated by f3tensor() */
void free_df3tensor(double ***t);
/* free a double f3tensor allocated by f3tensor() */

void mat_out_product_matrix3_permute_d3(double Q[27][27],
          const double a[3][3],
          const double b[3][3],
          const double c[3][3],
          const int permute[3]);
void mat_out_product_matrix3_d3(double Q[27][27],
                                const double a[3][3],
        const double b[3][3],
        const double c[3][3]);
void mat_transpose_matrix_d27(double a[27][27]);
int mat_check_zero_vector_d27(const double a[27], const double prec);
int mat_all_close_vector_d27(const double a[27], const double b[27], const double prec);
void mat_copy_vector_d27(double a[27], const double b[27]);
void mat_copy_mat_i27(int a[27][27], const int b[27][27]);
void mat_multiply_matrix_i27(int m[27][27],const int a[27][27],const int b[27][27]);
void mat_copy_mat_d27(double a[27][27], const double b[27][27]);
void mat_multiply_matrix_d27(double m[27][27],const double a[27][27],const double b[27][27]);
double mat_get_determinant_d3(const double a[3][3]);
int mat_get_determinant_i3(const int a[3][3]);
int mat_get_trace_i3( const int a[3][3] );
void mat_copy_array_i(int *a, const int *b, const int length);
void mat_copy_matrix_d3(double a[3][3], const double b[3][3]);
void mat_copy_matrix_i3(int a[3][3], const int b[3][3]);
void mat_copy_vector_d3(double a[3], const double b[3]);
void mat_copy_vector_i3(int a[3], const int b[3]);
void mat_add_vector_d3(double c[3], const double a[3], const double b[3]);
void mat_sub_vector_d3(double c[3], const double a[3], const double b[3]);
int mat_check_int_vector_d3(const double a[3], const double prec);
int mat_check_identity_matrix_i3(const int a[3][3],
         const int b[3][3]);
int mat_check_identity_matrix_d3( const double a[3][3],
          const double b[3][3],
          const double symprec );
int mat_check_identity_matrix_id3( const int a[3][3],
           const double b[3][3],
           const double symprec );
void mat_multiply_matrix_d3(double m[3][3],
          const double a[3][3],
          const double b[3][3]);
void mat_multiply_matrix_i3(int m[3][3],
          const int a[3][3],
          const int b[3][3]);
void mat_multiply_matrix_di3(double m[3][3],
           const double a[3][3],
           const int b[3][3]);
void mat_multiply_matrix_id3( double m[3][3],
            const int a[3][3],
            const double b[3][3]);
void mat_multiply_matrix_vector_i3(int v[3],
           const int a[3][3],
           const int b[3]);
void mat_multiply_matrix_vector_d3(double v[3],
           const double a[3][3],
           const double b[3]);
void mat_multiply_matrix_vector_id3(double v[3],
            const int a[3][3],
            const double b[3]);
void mat_multiply_matrix_vector_di3(double v[3],
            const double a[3][3],
            const int b[3]);
void mat_add_matrix_i3( int m[3][3],
      const int a[3][3],
      const int b[3][3] );
void mat_cast_matrix_3i_to_3d(double m[3][3],
            const int a[3][3]);
void mat_cast_matrix_3d_to_3i(int m[3][3],
            const double a[3][3]);
int mat_inverse_matrix_d3(double m[3][3],
        const double a[3][3],
        const double precision);
int mat_get_similar_matrix_d3(double m[3][3],
            const double a[3][3],
            const double b[3][3],
            const double precision);
void mat_transpose_matrix_d3(double a[3][3],
           const double b[3][3]);
void mat_transpose_matrix_i3(int a[3][3],
           const int b[3][3]);
void mat_get_metric( double metric[3][3],
         const double lattice[3][3]);
double mat_norm_squared_d3( const double a[3] );
int mat_norm_squared_i3( const int a[3] );
double mat_Dabs(const double a);
int mat_Nint(const double a);
double mat_Dmod1(const double a);
MatINT * mat_alloc_MatINT(const int size);
void mat_free_MatINT( MatINT * matint );
VecDBL * mat_alloc_VecDBL(const int size);
void mat_free_VecDBL( VecDBL * vecdbl );
Triplet * mat_alloc_Triplet(const int size);
void mat_free_Triplet( Triplet * triplet );

VecArbiLenINT *alloc_VecArbiLenINT(const int length);
void free_VecArbiLenINT(VecArbiLenINT *Vec);

VecArbiLenDBL *alloc_VecArbiLenDBL(const int length);
void free_VecArbiLenDBL(VecArbiLenDBL *Vec);

MatArbiLenINT *alloc_MatArbiLenINT(const int row, const int column);
void free_MatArbiLenINT(MatArbiLenINT *Mat);

MatArbiLenDBL *alloc_MatArbiLenDBL(const int row, const int column);
void free_MatArbiLenDBL(MatArbiLenDBL *Mat);

F3ArbiLenINT *alloc_F3ArbiLenINT(const int row, const int column, const int depth);
void free_F3ArbiLenINT(F3ArbiLenINT *F3);

F3ArbiLenDBL *alloc_F3ArbiLenDBL(const int row, const int column, const int depth);
void free_F3ArbiLenDBL(F3ArbiLenDBL *F3);
int mat_is_int_matrix( const double mat[3][3], double symprec );

int get_index_from_array(const int *a, const int length, const int value);
int get_index_from_vectors_i3(const int (*a)[3], const int length, const int value[3]);
int get_index_from_pos_vector_d3(const double *a[3], const int length, const double value[3], const double precesion);
int get_index_from_rots(const int (*a)[3][3], const int length, const int value[3][3]);
int array_unique(int *arr, const int *array_orig, const int length);
int  gaussian(double **b, int *IndexIndependent,double **a, const int row, const int column,const double prec);
#endif
