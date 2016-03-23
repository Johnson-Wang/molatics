#include "mathfunc.h"

VecArbiLenINT* get_all_symmetry_routes_to_star(const Symmetry *symmetry, const VecDBL *positions, const int atom, const int *mappings, const double symprec);

VecArbiLenINT* get_all_point_symmetry_routes_to_star(const PointSymmetry *ps, const VecDBL *positions, const int center, const int atom, const int *mappings, const double symprec);

int get_atom_sent_by_operation(const int orig_atom, const double (*positions)[3], const int rot[3][3], const double tran[3], const int num_atoms, const double precesion);

int get_atom_sent_by_rotation(const int orig_atom,const int center_atom,const double (*positions)[3], const int rot[3][3], const int num_atoms, const double precesion);