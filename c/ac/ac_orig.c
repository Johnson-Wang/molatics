#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ac_orig.h"
void auto_correlation_all_atoms(double* corr,
				const double* variable,
				const int num_atom,
				const int num_corr, 
				const int corr_length,
				const int sample_length)
{
  int i;
  double *corr_atom, *variable_atom;
  for (i=0;i<num_atom;i++)
  {
    corr_atom = corr + i * corr_length * 3 * 3;
    variable_atom = variable + i * num_corr *sample_length * 3;
    auto_correlation(corr_atom, variable_atom, num_corr, corr_length, sample_length);
  }
}

void auto_correlation(double* corr,
                      const double* variable,
                      const int num_corr, 
                      const int corr_length,
                      const int sample_length)
{
  int i, j,t,cor_time,p, m, n;
  double *v1,*v2;
//   for (i=0; i<10; i++)
//   {
//     for (j=0;j<3;j++)
//       printf("%20.7e", variable[0*sample_length*3+i*3+j]);
//     printf("\n");
//   }
#pragma omp parallel for collapse(2) private(cor_time, p, m, n, v1, v2)
  for (j=0;j<num_corr;j++)
  {
    for (t=0;t<corr_length;t++)
    {
      cor_time=sample_length-t;
      for (p=0;p<cor_time;p++)
      {
        v1 = variable + j * sample_length * 3 + p * 3;
        v2 = variable + j * sample_length * 3 + (p + t) * 3;
        for (m = 0; m < 3; m++)
        {
          for (n = 0; n < 3; n++)
          {
            #pragma omp atomic
              corr[t * 3 * 3 + m * 3 + n] += v1[m] * v2[n] / cor_time;
    //	    corr[t * 3 * 3 + m * 3 + n] += v1[m] * v2[n] / sample_length;
          }
        }
      }
    }
  }
  for (i = 0; i < corr_length * 3 * 3; i++)
  {
    corr[i] /= num_corr;
  }
}