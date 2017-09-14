#include<stdio.h>
#include<stdlib.h>
#include "mathfunc.h"
#include "force_constants.h"
#define LEN 10000
void rearrange_disp_fc2(double *ddcs, 
			const double *disps, 
			const double *coeff, 
			const double *trans,
			const int *ifcmap, 
			const int num_step,
			const int num_atom,
			const int num_irred, 
			const double coeff_cutoff)
{
  int a1, a2, a3, n, nt, i, j, k;
  const int aii = num_atom * 9 * 9;
  const int aj = num_atom * 3;
  const int ii = 9 * 9;
  double *c, *t, sum_temp;
  double coeff_temp[9][num_irred];
  int is_zero_coeff_temp[9][num_irred];
  for (a1=0; a1<num_atom; a1++)
  {
    for (a2=0; a2<num_atom; a2++)
    {
      for (i=0; i< 9; i++)
	for (j=0; j<num_irred; j++)
	{
	  coeff_temp[i][j] = 0;
	  is_zero_coeff_temp[i][j] = 1;
	}
      c = coeff + a1 * aii + a2 * ii;
      nt = ifcmap[a1 * num_atom + a2];
      t = trans + nt * 9 * num_irred;
      for (i=0; i<9; i++)
	for (k=0; k<num_irred; k++)
	  for (j=0; j<9; j++)
	    coeff_temp[i][k] += c[i*9+j] * t[j*num_irred + k];
      for (i=0; i< 9; i++)
	for (j=0; j<num_irred; j++)
	  if (mat_Dabs(coeff_temp[i][j]) > coeff_cutoff)
	    is_zero_coeff_temp[i][j] = 0;
      #pragma omp parallel for private(i, j, k, sum_temp)
      for (n=0; n<num_step; n++)
	for (i=0; i<3; i++)
	  for (j=0; j<3; j++)
	    for (k=0; k<num_irred; k++)
	    {
	      if (is_zero_coeff_temp[i*3+j][k])
		continue;
	      sum_temp = coeff_temp[i*3+j][k] * disps[n*aj+a2*3+j];
	      ddcs[n*aj*num_irred + a1*3*num_irred + i*num_irred+k] += sum_temp;
	    }
    }
  }
}
//
//void forces_from_fc2(double (*forces)[3],
//                    const double (*dist)[3],
//                    const double *fc2_reduced,
//                    const double *trans,
//                    const int *nindep,
//                    const int *coeff, // shape[natom, natom, natom]
//                    const int *fc2_map,  // shape[natom, natom, natom]
//                    const double (*tensor2)[9][9],
//                    const int num_step,
//                    const int num_atom,
//                    const int num_triplet){
//    int i, j, k, index, map, num_reduced=0;
//    double tr;
//    for (i = 0; i < num_triplet; i++)
//        num_reduced += nindep[i];
//    for (i = 0; i < num_step; i++){
//        for (j = 0; j < num_atom; j++){
//            for (k = 0; k < num_atom; k++)
//            {
//               index = coeff[j * num_atom + k];
//               tr = tensor2[index];
//               map = fc2_map[j*num_atom+k];
//
//               trans[]
//            }
//        }
//    }
//
//}
