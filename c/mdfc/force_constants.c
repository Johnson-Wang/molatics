#include <math.h>
#include "mathfunc.h"
VecArbiLenINT* get_all_symmetry_routes_to_star(const Symmetry *symmetry, const VecDBL *positions, const int atom, const int *mappings, const double symprec)
{
  VecArbiLenINT* operations = alloc_VecArbiLenINT(symmetry->size);
  int star, i;
  int num_opes=0;
  double pos[3], pos2[3], diff[3];
  init_ivector(operations->vec, operations->n, -1);
  star = mappings[atom];
  for (i=0; i< symmetry->size; i++)
  {
    mat_multiply_matrix_vector_id3(pos, symmetry->rot[i], positions->vec[atom]);
    mat_add_vector_d3(pos2, pos, symmetry->trans[i]);
    mat_sub_vector_d3(diff, pos2, positions->vec[star]);
    if (mat_check_int_vector_d3(diff, symprec))
    {
      operations->vec[num_opes++] = i;
    }
  }
  operations->n=num_opes;

  return operations;
}

VecArbiLenINT* get_all_point_symmetry_routes_to_star(const PointSymmetry *ps, const VecDBL *positions, const int center, const int atom, const int *mappings, const double symprec)
{
  VecArbiLenINT* operations = alloc_VecArbiLenINT(ps->size);
  int star, i, num_opes=0;
  double pos[3], pos2[3], diff[3], rel[3];
  init_ivector(operations->vec, operations->n, -1);
  star = mappings[atom];
  for (i=0; i< ps->size; i++)
  {
    mat_sub_vector_d3(rel, positions->vec[atom], positions->vec[center]);
    mat_multiply_matrix_vector_id3(pos, ps->rot[i], rel);
    mat_add_vector_d3(pos2, pos, positions->vec[center]);
    mat_sub_vector_d3(diff, pos2, positions->vec[star]);
    if (mat_check_int_vector_d3(diff, symprec))
      operations->vec[num_opes++] = i;
  }
  operations->n=num_opes;
  return operations;
}

int get_atom_sent_by_operation(const int orig_atom, const double (*positions)[3], const int rot[3][3], const double tran[3], const int num_atoms, const double precesion)
{
  double rot_pos[3], ope_pos[3], diff[3];
  int i;
  mat_multiply_matrix_vector_id3(rot_pos, rot, positions[orig_atom]);
  mat_add_vector_d3(ope_pos, rot_pos, tran);
  for (i=0; i<num_atoms; i++)
  {
    mat_sub_vector_d3(diff, positions[i], ope_pos);
    if (mat_Dabs(mat_Nint(diff[0]) - diff[0]) < precesion &&
        mat_Dabs(mat_Nint(diff[1]) - diff[1]) < precesion &&
        mat_Dabs(mat_Nint(diff[2]) - diff[2]) < precesion)
      return i;
  }
  return -1;
}

int get_atom_sent_by_rotation(const int orig_atom, const int center_atom, const double (*positions)[3], const int rot[3][3], const int num_atoms, const double precesion)
{
  double rot_pos[3], ope_pos[3], diff[3], pos[3];
  int i;
  mat_sub_vector_d3(pos, positions[orig_atom], positions[center_atom]);
  mat_multiply_matrix_vector_id3(rot_pos, rot, pos);
  mat_add_vector_d3(ope_pos, rot_pos, positions[center_atom]);
  for (i=0; i<num_atoms; i++)
  {
    mat_sub_vector_d3(diff, positions[i], ope_pos);
    if (mat_Dabs(mat_Nint(diff[0]) - diff[0]) < precesion &&
        mat_Dabs(mat_Nint(diff[1]) - diff[1]) < precesion &&
        mat_Dabs(mat_Nint(diff[2]) - diff[2]) < precesion)
      return i;
  }
  return -1;
}