#include<stdio.h>
#include<stdlib.h>
#include "mathfunc.h"
#include "force_constants.h"
#define LEN 10000

int permute3[6][3] = {{0, 1, 2},{0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

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
                                    const double symprec)
{
  F3ArbiLenDBL* transform;
  VecArbiLenINT *operations;
  VecArbiLenINT *rotations2;
  VecArbiLenINT *rotations3;
  PointSymmetry *ps2, *ps3;
  PointSymmetry *ps_triplets = sym_alloc_point_symmetry(LEN);
  int num_atoms = positions->size;
  int *unique_atoms1 = ivector(num_atoms);
  int *unique_atoms2 = ivector(num_atoms);
  int *unique_atoms3 = ivector(num_atoms);
  int *mapping2 = ivector(num_atoms);
  int *mapping3 = ivector(num_atoms);
  double ***transform_temp = df3tensor(triplets->size,27,27);
  VecArbiLenINT *NIndependent = alloc_VecArbiLenINT(triplets->size);
  MatArbiLenINT *IndexIndependent = alloc_MatArbiLenINT(triplets->size, 27);
  MatArbiLenDBL *CC= alloc_MatArbiLenDBL(LEN, 27);
  int atom1, atom2, atom3, atom1_0, atom2_0, atom3_0, atom1_1, atom2_1, atom3_1, atom2_2, atom3_2;
  int num_unique1, num_unique2;
  int itriplet, ipermute, i, j, k, l, m, index1, index2, permute_temp[3];
  int rot1[3][3], rot2[3][3], rot3[3][3], rot_temp[3][3], rot[3][3]; 
  double trans1[3];
  double rot_db[3][3], rot_cart[3][3];
  double PP[27][27];
  int num_ind=0, is_found=0;
  num_unique1 = array_unique(unique_atoms1, mappings1->vec, num_atoms);

  for(itriplet = 0; itriplet < triplets->size; itriplet++)
  {
    init_dmatrix(CC->mat, LEN, CC->column, 0.0); CC->row=1; // init CC with 1 to guarantee that an array is sent to the gaussian elimination module
    atom1 = triplets->tri[itriplet][0];
    atom2 = triplets->tri[itriplet][1];
    atom3 = triplets->tri[itriplet][2];
    for (ipermute = 0; ipermute < 6; ipermute++)
    {
      ps_triplets->size=0;
      atom1_0 = triplets->tri[itriplet][permute3[ipermute][0]];
      atom2_0 = triplets->tri[itriplet][permute3[ipermute][1]];
      atom3_0 = triplets->tri[itriplet][permute3[ipermute][2]];
      atom1_1 = mappings1->vec[atom1_0];
      index1=get_index_from_array(unique_atoms1, num_unique1, atom1_1);
      if (atom1_1 != atom1) continue;
      operations = get_all_symmetry_routes_to_star(symmetry1, positions, atom1_0, mappings1->vec, symprec);
      for (i=0; i<operations->n; i++)
      {
        mat_copy_matrix_i3(rot1, symmetry1->rot[operations->vec[i]]);
        mat_copy_vector_d3(trans1, symmetry1->trans[operations->vec[i]]);
        atom2_1 = get_atom_sent_by_operation(atom2_0, positions->vec, rot1, trans1, num_atoms, symprec);
        atom3_1 = get_atom_sent_by_operation(atom3_0, positions->vec, rot1, trans1, num_atoms, symprec);
        ps2 = (PointSymmetry*)pointsymmetry2 + index1;
        mat_copy_array_i(mapping2, mappings2->mat[index1], num_atoms);
        num_unique2 = array_unique(unique_atoms2, mapping2, num_atoms);
        if (mapping2[atom2_1] != atom2) continue;
        atom2_2 = mapping2[atom2_1];
        index2 = get_index_from_array(unique_atoms2, num_unique2, atom2_2);
        rotations2 = get_all_point_symmetry_routes_to_star(ps2, positions, atom1_1,  atom2_1, mapping2, symprec);
	
        for (j=0; j<rotations2->n; j++)
        {
          mat_copy_matrix_i3(rot2, ps2->rot[rotations2->vec[j]]);
          atom3_2 = get_atom_sent_by_rotation(atom3_1, atom1_1, positions->vec, rot2, num_atoms, symprec);
          ps3 = (PointSymmetry*)pointsymmetry3 + index1 * mappings3->column + index2;   // the column of mapping3 is equal to the max unique_atoms2 
          mat_copy_array_i(mapping3, mappings3->f3[index1][index2], num_atoms);
          if (mapping3[atom3_2] != atom3) continue;
          rotations3 = get_all_point_symmetry_routes_to_star(ps3, positions, atom2_2,  atom3_2, mapping3, symprec);

          for (k=0; k<rotations3->n; k++)
          {
            mat_copy_matrix_i3(rot3, ps3->rot[rotations3->vec[k]]);
            mat_multiply_matrix_i3(rot_temp, rot2, rot1);
            mat_multiply_matrix_i3(rot, rot3, rot_temp);
            if(get_index_from_rots(ps_triplets->rot, ps_triplets->size, rot) != -1) continue; // skip repeated rotational components to improve the speed
            mat_copy_matrix_i3(ps_triplets->rot[ps_triplets->size++], rot); 
            mat_cast_matrix_3i_to_3d(rot_db, rot);
            mat_get_similar_matrix_d3(rot_cart, rot_db, lattice, 0);
            mat_transpose_matrix_d3(rot_cart, rot_cart); // Equal to inverse, meaning the rot_cart now maps the star triplet to the original one
            mat_kron_product_matrix3_d3(PP, rot_cart, rot_cart, rot_cart);
            for (l = 0; l < 3; l++)
                permute_temp[l] = get_index_from_array(permute3[ipermute], 3, l);  // find the reverse permutation
            mat_permute_d27(PP, permute_temp, 0);
            for (l=0; l<27; l++)
              PP[l][l] -= 1.0;
            for (l = 0; l < 27; l++)
            {
              is_found = 0;
              if (mat_check_zero_vector_d27(PP[l], symprec)) continue;
              mat_normalize_by_abs_vector_d27(PP[l], symprec);
              for (m = 0; m < CC->row; m++)
                if (mat_all_close_vector_d27(PP[l], CC->mat[m], symprec))
                {
                  is_found = 1;
                  break;
                }
              if (!is_found)
              {
                mat_copy_vector_d27(CC->mat[CC->row], PP[l]);
                (CC->row)++;
                if (CC->row>=LEN)
                  runerror("Size of transformation matrix CC not enough");
              }
            }
          }
          
          free_VecArbiLenINT(rotations3);
        }
        free_VecArbiLenINT(rotations2);
      }
    }

    NIndependent->vec[itriplet] = gaussian(transform_temp[itriplet], IndexIndependent->mat[itriplet], CC->mat, CC->row, 27, symprec);
    
    num_ind += NIndependent->vec[itriplet];  
  }
  
  
  transform = alloc_F3ArbiLenDBL(triplets->size, 27, num_ind);
  init_df3tensor(transform->f3, triplets->size, 27, num_ind, 0.0);
  num_ind = 0;
  for (itriplet=0; itriplet<triplets->size; itriplet++)
  {
    for (i=0; i < NIndependent->vec[itriplet]; i++)
    {
      for (j=0; j<27; j++)
          transform->f3[itriplet][j][num_ind + i] = transform_temp[itriplet][j][i];
      Independents[num_ind + i] = IndexIndependent->mat[itriplet][i] + itriplet * 27;
    }
    num_ind += NIndependent->vec[itriplet];
  }
  free_ivector(unique_atoms1);
  free_ivector(unique_atoms2);
  free_ivector(unique_atoms3);
  free_ivector(mapping2);
  free_ivector(mapping3);
  free_df3tensor(transform_temp); 
  free_VecArbiLenINT(operations);
  free_VecArbiLenINT(NIndependent);
  free_MatArbiLenINT(IndexIndependent);
  free_MatArbiLenDBL(CC);
  sym_free_point_symmetry(ps_triplets);
  return transform;
}

void get_fc3_coefficients(double (*coefficients)[27][27], 
                          int *ifc_mapping,
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
                          const double symprec)
{
  PointSymmetry *ps2, *ps3;
  int natoms = positions->size;
  int ntriplets = triplets->size;
  int *unique_atoms1 = ivector(natoms);
  int *unique_atoms2 = ivector(natoms);
  int *unique_atoms3 = ivector(natoms);
  int *unique_triplets = ivector(ntriplets);
  int *mapping2 = ivector(natoms);
  int *mapping3 = ivector(natoms);
  int a1, a2, a3, atom1_1, atom2_1, atom3_1, atom2_2, atom3_2, atom3_3;
  int num_unique1, num_unique2, num_uniquet;
  int index1, index2, indext, indexi;
  int rot1[3][3], rot2[3][3], rot3[3][3], rot_temp[3][3], rot[3][3]; 
  double trans1[3];
  int triplet_rot[3];
  double rot_db[3][3], rot_cart[3][3];
  double PP[27][27];
  num_unique1 = array_unique(unique_atoms1, mappings1->vec, natoms);
  num_uniquet = array_unique(unique_triplets, triplet_mapping, ntriplets);
  
  for(a1=0; a1<natoms; a1++)
  {
    atom1_1 = mappings1->vec[a1];
    index1=get_index_from_array(unique_atoms1, num_unique1, atom1_1);
//     printf("atom1_0:%3d\n", a1);
    mat_copy_matrix_i3(rot1, symmetry1->rot[mapope1->vec[a1]]);
//     printf("mapope1:%3d\n", mapope1->vec[a1]);
    mat_copy_vector_d3(trans1, symmetry1->trans[mapope1->vec[a1]]);
    ps2 = (PointSymmetry*)pointsymmetry2 + index1;
    mat_copy_array_i(mapping2, mappings2->mat[index1], natoms);
    num_unique2 = array_unique(unique_atoms2, mapping2, natoms);
    for (a2=0; a2<natoms; a2++)
    {
      atom2_1 = get_atom_sent_by_operation(a2, positions->vec, rot1, trans1, natoms, symprec);
      atom2_2 = mapping2[atom2_1];
//       printf("atom2_0:%3d\n", a2);
      index2 = get_index_from_array(unique_atoms2, num_unique2, atom2_2);
      mat_copy_matrix_i3(rot2, ps2->rot[mapope2->mat[index1][atom2_1]]);
      ps3 = (PointSymmetry*)pointsymmetry3 + index1 * mappings3->column + index2;   // the column of mapping3 is equal to the max unique_atoms2 
      for (a3=0; a3<natoms; a3++)
      {
        atom3_1 = get_atom_sent_by_operation(a3, positions->vec, rot1, trans1, natoms, symprec);
        atom3_2 = get_atom_sent_by_rotation(atom3_1, atom1_1, positions->vec, rot2, natoms, symprec);
        mat_copy_array_i(mapping3, mappings3->f3[index1][index2], natoms);
        atom3_3 = mapping3[atom3_2];
        mat_copy_matrix_i3(rot3, ps3->rot[mapope3->f3[index1][index2][atom3_2]]);
        mat_multiply_matrix_i3(rot_temp, rot2, rot1);
        mat_multiply_matrix_i3(rot, rot3, rot_temp);
        mat_cast_matrix_3i_to_3d(rot_db, rot);
        mat_get_similar_matrix_d3(rot_cart, rot_db, lattice, 0); // from a general atom to its star
        mat_transpose_matrix_d3(rot_cart, rot_cart); // from a star to its orbitals
        mat_kron_product_matrix3_d3(PP, rot_cart, rot_cart, rot_cart);
        triplet_rot[0] = atom1_1; triplet_rot[1] = atom2_2; triplet_rot[2] = atom3_3;
        indext = get_index_from_vectors_i3(triplets->tri, triplets->size, triplet_rot); // index of triplet
        indexi = get_index_from_array(unique_triplets, num_uniquet, triplet_mapping[indext]); // index of irreducible triplet

        mat_multiply_matrix_d27(PP, PP, triplet_transform[indext]);
        ifc_mapping[a1 * natoms * natoms + a2 * natoms + a3] = indexi;
        mat_copy_mat_d27(coefficients[a1 * natoms * natoms + a2 * natoms + a3], PP);
      }
    }
  }
  free_ivector(unique_atoms1);
  free_ivector(unique_atoms2);
  free_ivector(unique_atoms3);
  free_ivector(unique_triplets);
  free_ivector(mapping2);
  free_ivector(mapping3);
}



void rearrange_disp_fc3(double *ddcs, 
			const double *disps, 
			const double *coeff, 
			const double *trans,
			const int *ifcmap, 
			const int num_step,
			const int num_atom,
			const int num_irred, 
			const double coeff_cutoff)
{
  int a1, a2, a3, n, nt, i, j, k, l;
  const int aa = num_atom * num_atom;
  const int aaii = num_atom * num_atom * 27 * 27;
  const int aii = num_atom * 27 * 27;
  const int aj = num_atom * 3;
  const int ii = 27 * 27;
  double *c, *t, sum_temp;
  double coeff_temp[27][num_irred];
  int is_zero_coeff_temp[27][num_irred];
  for (a1=0; a1<num_atom; a1++)
  {
    for (a2=0; a2<num_atom; a2++)
    {
      for (a3=a2; a3<num_atom; a3++) // Considering the permuation symmetry
      {
	for (i=0; i< 27; i++)
	  for (j=0; j<num_irred; j++)
	  {
	    coeff_temp[i][j] = 0;
	    is_zero_coeff_temp[i][j] = 1;
	  }
	c = coeff + a1 * aaii + a2 * aii + a3 * ii;
	nt = ifcmap[a1 * aa + a2 * num_atom + a3];
	t = trans + nt * 27 * num_irred;
	for (i=0; i<27; i++)
	  for (k=0; k<num_irred; k++)
	    for (j=0; j<27; j++)
	      coeff_temp[i][k] += c[i*27+j] * t[j*num_irred + k];
	for (i=0; i< 27; i++)
	  for (j=0; j<num_irred; j++)
	    if (mat_Dabs(coeff_temp[i][j]) > coeff_cutoff)
	      is_zero_coeff_temp[i][j] = 0;
        #pragma omp parallel for private(i, j, k, l, sum_temp)
	for (n=0; n<num_step; n++)
	  for (i=0; i<3; i++)
	    for (j=0; j<3; j++)
	      for (k=0; k<3; k++)
		for (l=0; l<num_irred; l++)
		{
		  if (is_zero_coeff_temp[i*9+j*3+k][l])
		    continue;
		  sum_temp = coeff_temp[i*9+j*3+k][l] * disps[n*aj+a2*3+j] * disps[n*aj+a3*3+k];
		  if (a3 != a2) sum_temp *= 2; 
		  ddcs[n*aj*num_irred + a1*3*num_irred + i*num_irred+l] += sum_temp;
		}
      }
    }
  }
}