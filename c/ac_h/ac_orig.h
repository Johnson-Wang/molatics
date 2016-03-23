void auto_correlation(double* corr,
                      const double* variable,
                      const int num_corr, 
                      const int corr_length,
                      const int sample_length);

void auto_correlation_all_atoms(double* corr,
				const double* variable,
				const int num_atom,
				const int num_corr, 
				const int corr_length,
				const int sample_length);