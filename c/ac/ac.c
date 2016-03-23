#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ac.h"
void auto_correlation_all_atoms(double* corr,
				const double* variable,
				const int num_atom,
				const int num_corr, 
				const int corr_length,
				const int sample_length,
				const int dim)
{
  int i;
  double *corr_atom, *variable_atom;
  for (i=0;i<num_atom;i++)
  {
    corr_atom = corr + i * corr_length * dim * dim;
    variable_atom = variable + i * num_corr *sample_length * dim;
    auto_correlation(corr_atom, variable_atom, num_corr, corr_length, sample_length, dim);
  }
}

void auto_correlation(double* corr,
                      const double* variable,
                      const int num_corr, 
                      const int corr_length,
                      const int sample_length, 
		      const int dim)
{
  int i, j,t,cor_time,p, m, n;
  double *v1,*v2;
#pragma omp parallel for collapse(2) private(cor_time, p, m, n, v1, v2)
  for (j=0;j<num_corr;j++)
  {
    for (t=0;t<corr_length;t++)
    {
      cor_time=sample_length-t;
      for (p=0;p<cor_time;p++)
      {
        v1 = variable + j * sample_length * dim + p * dim;
        v2 = variable + j * sample_length * dim + (p + t) * dim;
        for (m = 0; m < dim; m++)
        {
          for (n = 0; n < dim; n++)
          {
            #pragma omp atomic
              corr[t * dim * dim + m * dim + n] += v1[m] * v2[n] / cor_time;
    //	    corr[t * dim * dim + m * dim + n] += v1[m] * v2[n] / sample_length;
          }
        }
      }
    }
  }
  for (i = 0; i < corr_length * dim * dim; i++)
  {
    corr[i] /= num_corr;
  }
}