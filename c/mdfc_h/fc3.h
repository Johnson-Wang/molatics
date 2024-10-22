#include "mathfunc.h"
F3ArbiLenDBL* get_fc3_spg_invariance(int *Independents,
                                    const Triplet *triplets,
                                    const VecDBL *positions,
                                    const Symmetry *symmetry1,
                                    const VecArbiLenINT *mappings1,
                                    const PointSymmetry *pointsymmetry2,
                                    const MatArbiLenINT *mappings2,
                                    const PointSymmetry *pointsymmetry3,
                                    const F3ArbiLenINT *mappings3,
                                    const double lattice[3][3],
                                    const double symprec);
void get_fc3_coefficients(double (*coefficients)[27][27], 
                          int *ifc_mapping,
                          const VecArbiLenINT *first_atoms,
                          const Triplet *triplets,
                          const int *triplet_mapping,
                          const double (*triplet_transform)[27][27],
                          const double lattice[3][3],
                          const VecDBL *positions,
                          const Symmetry *symmetry1,
                          const VecArbiLenINT *mappings1,
                          const VecArbiLenINT *mapope1,
                          const PointSymmetry *pointsymmetry2,
                          const MatArbiLenINT *mappings2,
                          const MatArbiLenINT *mapope2,
                          const PointSymmetry *pointsymmetry3,
                          const F3ArbiLenINT *mappings3,
                          const F3ArbiLenINT *mapope3,
                          const double symprec);
int get_fc3_coefficients_triplet(double coefficients[27][27],
                                  const int triplet[3],
                                  const Triplet *triplets,
                                  const int *triplet_mapping,
                                  const double (*triplet_transform)[27][27],
                                  const double lattice[3][3],
                                  const VecDBL *positions,
                                  const Symmetry *symmetry1,
                                  const VecArbiLenINT *mappings1,
                                  const VecArbiLenINT *mapope1,
                                  const PointSymmetry *pointsymmetry2,
                                  const MatArbiLenINT *mappings2,
                                  const MatArbiLenINT *mapope2,
                                  const PointSymmetry *pointsymmetry3,
                                  const F3ArbiLenINT *mappings3,
                                  const F3ArbiLenINT *mapope3,
                                  const double symprec);
void rearrange_disp_fc3(double *ddcs, 
			const double *disps, 
			const double *coeff, 
			const double *trans,
			const int *ifcmap, 
			const int num_step,
			const int num_atom,
			const int num_irred,
			const double coeff_cutoff); // the fc3 coefficient tolerance 