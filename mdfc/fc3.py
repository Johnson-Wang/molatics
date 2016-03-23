__author__ = 'xinjiang'
import numpy as np
from fc2 import get_pairs, get_next_atom, get_atom_sent_by_operation, get_atom_sent_by_site_sym,\
    get_all_operations_at_star, get_rotations_at_star, gaussian, get_operations_at_star
from phonopy.harmonic.force_constants import similarity_transformation
from itertools import permutations
from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors

def get_bond_symmetry(site_symmetry,
                      positions,
                      atom_center,
                      atom_disp,
                      symprec=1e-5):
    """
    Bond symmetry is the symmetry operations that keep the symmetry
    of the cell containing two fixed atoms.
    """
    bond_sym = []
    pos = positions
    for rot in site_symmetry:
        rot_pos = (np.dot(pos[atom_disp] - pos[atom_center], rot.T) +
                   pos[atom_center])
        diff = pos[atom_disp] - rot_pos
        if (abs(diff - diff.round()) < symprec).all():
            bond_sym.append(rot)

    return np.array(bond_sym)

def get_irreducible_triplets_with_permute(triplets,
                                          positions,
                                          rotations1,
                                          translations1,
                                          mappings1,
                                          mapping_opes1,
                                          rotations2,
                                          num_rotations2,
                                          mappings2,
                                          mapping_opes2,
                                          rotations3,
                                          num_rotations3,
                                          mappings3,
                                          mapping_opes3,
                                          lattice,
                                          symprec):
    triplets_mapping = np.arange(len(triplets))
    transform = np.array([np.eye(27) for i in  np.arange(len(triplets))], dtype=np.float)
    indatoms1 = np.unique(mappings1)
    for itriplet, (p1, p2, p3) in enumerate(triplets):
        triplet = (p1, p2, p3)
        for permute in list(permutations([0,1,2]))[1:]: # Leave out the original triplet
            atom1, atom2, atom3 = [triplet[i] for i in permute] # permuted triplet
            if triplet == (atom1, atom2, atom3):
                continue
                # atom1 = triplet[permute_matrix[0]];atom2 = triplet[permute_matrix[1]]; atom3 = triplet[permute_matrix[2]] # permuted triplet
            numope = mapping_opes1[atom1] #number of symmetry operation
            rot, tran = rotations1[numope], translations1[numope]
            atom1 = mappings1[atom1]
            atom2 = get_atom_sent_by_operation(atom2, positions, rot, tran, symprec=symprec)
            atom3 = get_atom_sent_by_operation(atom3, positions, rot, tran, symprec=symprec)
            # atom1 is transformed to an independent atom and atom2 and atom3 are transformed following atom1
            index1 = np.where(indatoms1 == atom1)[0][0] # index of the first irreducible atom
            site_sym = rotations2[index1][:num_rotations2[index1]]
            mapping2 = mappings2[index1] # now atom1 would be fixed under its site symmetries
            iatoms2 = np.unique(mappings2[index1])
            rot2 = site_sym[mapping_opes2[index1,atom2]]
            # a rotation which transforms atom2 to an independent one under site symmetries
            atom2 = mapping2[atom2]
            atom3 = get_atom_sent_by_site_sym(atom3, atom1, positions, rot2, symprec=symprec)
            # atom3 is transformed following atom2
            index2 = np.where(iatoms2 == atom2)[0][0]# index of the second irreducible atom
            site_sym3 = rotations3[index1, index2, :num_rotations3[index1, index2]]
            map_ope3 = mapping_opes3[index1,index2]
            mapping3 = mappings3[index1, index2]
            rot3 = site_sym3[map_ope3[atom3]]
            atom3 = mapping3[atom3] # atom3 is also transformed to an irreducible one under bond symmetries
            triplet_permute = (atom1, atom2, atom3)
            # Check if triplet_permute already exists in the triplets pool
            permuted_index = triplets.index(triplet_permute)
            if triplets_mapping[permuted_index] < triplets_mapping[itriplet]:
                triplets_mapping[itriplet] = triplets_mapping[permuted_index]
                transf = np.dot(rot3, np.dot(rot2, rot))
                transf_cart = similarity_transformation(lattice, transf)
                seq = "".join(["ikm"[i] for i in permute])
                transform_temp = np.einsum("ij,kl,mn -> %sjln"%seq,
                                                     transf_cart, transf_cart, transf_cart).reshape(27,27)
                transform[itriplet] = np.dot(transform[permuted_index], transform_temp)
                # from the general one to irreducible one
    return triplets_mapping, transform

def get_fc3_coefficients(triplets_orig,
                         mapping_triplet,
                         transform_triplet,
                         lattice,
                         positions,
                         rotations1,
                         translations1,
                         mappings1,
                         map_ope1,
                         rotations2,
                         mappings2,
                         map_ope2,
                         rotations3,
                         mappings3,
                         map_ope3,
                         symprec):
    natom = len(positions)
    ind_atoms1 = np.unique(mappings1)
    triplets_reduced = [triplets_orig[i] for i in np.unique(mapping_triplet)]
    ifc_map = np.zeros((natom, natom, natom), dtype=np.int16)
    coeff = np.zeros((natom, natom, natom, 27, 27), dtype=np.float)

    for atom1 in np.arange(natom):
        iratom1 = mappings1[atom1]
        numope = map_ope1[atom1]
        index1 = np.where(ind_atoms1 == iratom1)[0][0]
        ind_atoms2 = np.unique(mappings2[index1])
        rot1, tran1 = rotations1[numope], translations1[numope]
        for atom2 in np.arange(natom):
            iratom2 = get_atom_sent_by_operation(atom2, positions, rot1, tran1, symprec=symprec)
            rot2 = rotations2[index1, map_ope2[index1, iratom2]]
            iratom2 = mappings2[index1, iratom2]
            index2 = np.where(ind_atoms2 == iratom2)[0][0]
            for atom3 in np.arange(natom):
                if atom1 == 4 and atom2 == 4 and atom3 == 4:
                    print
                iratom3 = get_atom_sent_by_operation(atom3, positions, rot1, tran1, symprec=symprec)
                iratom3 = get_atom_sent_by_site_sym(iratom3, iratom1, positions, rot2, symprec=symprec)
                rot3 = rotations3[index1, index2, map_ope3[index1, index2, iratom3]]
                iratom3 =mappings3[index1, index2, iratom3]
                rot = np.dot(rot3, np.dot(rot2, rot1))
                #rot transforms an arbitrary triplet to the irreducible one
                index_triplet = triplets_orig.index((iratom1, iratom2, iratom3))
                star3 = triplets_orig[mapping_triplet[index_triplet]]
                rot_cart = np.double(similarity_transformation(lattice, rot))
                coeff_temp = np.einsum("ij, kl, mn -> ikmjln", rot_cart, rot_cart, rot_cart).reshape(27,27)
                ifc_map[atom1, atom2, atom3] = triplets_reduced.index(star3)
                coeff_temp = np.dot(transform_triplet[index_triplet], coeff_temp)
                #take the permutation matrix into consideration
                coeff[atom1, atom2, atom3] = coeff_temp.T # inverse equals transpose
    return  coeff, ifc_map

def get_fc3_spg_invariance(triplets,
                           is_triplets_included,
                           positions,
                           rotations1,
                           translations1,
                           mappings1,
                           rotations2,
                           num_rotations2,
                           mappings2,
                           rotations3,
                           num_rotations3,
                           mappings3,
                           lattice,
                           symprec):
    triplets_dict = []
    iatoms1 = np.unique(mappings1)
    for itriplet, triplet in enumerate(triplets):
        triplet_dict = {}
        triplet_dict["triplet"] = triplet
        CC = [np.zeros(27)]
        if not is_triplets_included[itriplet]:
            triplet_dict['independent'] = [] # independent ele
            triplet_dict['transform'] = None
            triplets_dict.append(triplet_dict)
            continue
        for permute in list(permutations([0,1,2])): # Leave out the original triplet
            rot_all = []
            atom1, atom2, atom3 = [triplet[i] for i in permute]
            if not mappings1[atom1] == triplet[0]:
                continue
            for numope in get_all_operations_at_star(rotations1,
                                                     translations1,
                                                     positions,
                                                     atom1,
                                                     mappings1,
                                                     symprec):
                #get all the symmetry operation that keeps atom1 unchanged
                rot1, tran = rotations1[numope], translations1[numope]
                atom1_1 = mappings1[atom1]
                index1 = np.where(iatoms1 == atom1_1)[0][0] # index of the first irreducible atom
                atom2_1 = get_atom_sent_by_operation(atom2, positions, rot1, tran, symprec=symprec)
                atom3_1 = get_atom_sent_by_operation(atom3, positions, rot1, tran, symprec=symprec)
                site_syms = rotations2[index1, :num_rotations2[index1]]
                mapping2 = mappings2[index1] # now atom1 would be fixed under its site symmetries
                iatoms2 = np.unique(mapping2)
                if mapping2[atom2_1] != triplet[1]:
                    continue
                for rot2 in get_rotations_at_star(site_syms, positions, atom1_1, atom2_1, mapping2, symprec):
                    atom2_2 = mapping2[atom2_1]
                    atom3_2 = get_atom_sent_by_site_sym(atom3_1, atom1_1, positions, rot2, symprec=symprec) #no matter atom1_1 or atom2_1
                    index2 = np.where(iatoms2 == atom2_2)[0][0]# index of the second irreducible atom
                    site_syms3 = rotations3[index1, index2, :num_rotations3[index1, index2]]
                    mapping3 = mappings3[index1, index2]
                    if not mapping3[atom3_2] == triplet[2]:
                        continue
                    for rot3 in get_rotations_at_star(site_syms3, positions, atom2_2, atom3_2, mapping3, symprec):
                        rot = np.dot(rot3, np.dot(rot2, rot1))
                        isfound = False
                        for r in rot_all:
                            if (np.abs(r-rot) < 1e-8).all():
                                isfound = True
                                break
                        if not isfound:
                            rot_all.append(rot)
                        else:
                            continue
                        rot_cart = np.double(similarity_transformation(lattice, rot))
                        seq = "".join(["ikm"[i] for i in permute])
                        PP  = np.einsum("ij,kl,mn -> %sjln"%seq, rot_cart, rot_cart, rot_cart).reshape(27, 27).T
                        BB = PP - np.eye(27)
                        for i in np.arange(27):
                            is_found = False
                            if not (np.abs(BB[i]) < 1e-8).all():
                                for j in np.arange(len(CC)):
                                    if (np.abs(BB[i] - CC[j]) < 1e-8).all():
                                        is_found = True
                                        break
                                if not is_found:
                                    CC.append(BB[i])
        CC, transform, independent = gaussian(np.array(CC))
        # if (np.abs(np.dot(transform, fc3[triplet[0], triplet[1], triplet[2]].flatten()[independent]) -
        #         fc3[triplet[0], triplet[1], triplet[2]].flatten()) > 1e-3).any():
        #     print triplet
        triplet_dict['independent'] = [ind + itriplet * 27 for ind in independent] # independent ele
        triplet_dict['transform'] = transform
        triplets_dict.append(triplet_dict)
    num_irred = [len(dic['independent']) for dic in triplets_dict]
    trans = np.zeros((len(triplets_dict), 27, sum(num_irred)), dtype="float")
    for i, trip  in enumerate(triplets_dict):
        start = sum(num_irred[:i])
        length = num_irred[i]
        trans[i,:, start:start + length] = trip['transform']
    ind_elements =  np.hstack((dic['independent'] for dic in triplets_dict)).astype("intc")
    return ind_elements, trans

def get_fc3_translational_invariance(trans,
                                     irred_ele,
                                     coeff,
                                     ifc_map,
                                     indatoms1,
                                     indatoms2,
                                     nindatoms2,
                                     positions):
    # set translational invariance
    num_irred = len(irred_ele)
    natom = len(positions)
    ti_transforms =[]
    for i, atom1 in enumerate(indatoms1):
        iatoms2 = indatoms2[i, :nindatoms2[i]]
        for j, atom2 in enumerate(iatoms2):
            ti_transform = np.zeros((27, num_irred))
            for atom3 in range(natom):
                irred_triplet = ifc_map[atom1, atom2, atom3]
                transform = trans[irred_triplet]
                ti_transform += np.dot(coeff[atom1, atom2, atom3], transform)
            for k in range(27):
                if not (np.abs(ti_transform[k])< 1e-4).all():
                    ti_transforms.append(ti_transform[k])
    print "Number of constraints of fc3 from translational invariance:%d"%len(ti_transforms)
    CC, transform, independent = gaussian(np.array(ti_transforms))
    new_irred_ele = irred_ele[independent]
    trans2 = np.dot(trans, transform)
    return new_irred_ele, trans2

def get_fc3_rotational_invariance(fc2, trans, irred_elements, coeff, ifc_map, indatoms1, indatoms2, nindatoms2, cell, symprec):
    positions = cell.get_scaled_positions()
    natom = len(positions)
    num_irred2 = trans.shape[-1]
    lattice = cell.get_cell()
    eijk = np.zeros((3,3,3), dtype="intc")
    eijk[0,1,2] = eijk[1,2,0] = eijk[2,0,1] = 1
    eijk[0,2,1] = eijk[2,1,0] = eijk[1,0,2] = -1 # epsilon matrix, which is an antisymmetric 3 * 3 * 3 tensor
    torques =[np.zeros(num_irred2)]
    for i, atom1 in enumerate(indatoms1):
        iatoms2 = indatoms2[i, :nindatoms2[i]]
        for j, atom2 in enumerate(iatoms2):
            torque = np.zeros((27, num_irred2), dtype=np.float)
            for atom3 in range(natom):
                fc_temp = np.dot(coeff[atom1, atom2, atom3], trans[ifc_map[atom1, atom2, atom3]]).reshape(3,3,3, -1)
                vectors = get_equivalent_smallest_vectors(atom3,
                                                          atom1,
                                                          cell,
                                                          lattice,
                                                          symprec)
                r_frac = np.array(vectors).sum(axis=0) / len(vectors)
                r = np.dot(r_frac, lattice)
                disp = np.einsum("j, ijk -> ik", r, eijk)
                t3 = np.einsum("abcN, cv -> abvN", fc_temp, disp).reshape(27, num_irred2)
                t2_1 = np.einsum("bc, cav -> abv", fc2[atom1, atom2], eijk).flatten()
                t2_2 = np.einsum("ac, cbv -> abv", fc2[atom1, atom2], eijk).flatten()
                torque += (t3.T + t2_1 + t2_2).T
            for k in range(27):
                if not (np.abs(torque[k])< 1e-5).all():
                    torques.append(torque[k])
    print "Number of constraints of fc3 from rotational invariance:%d"%len(torques)
    CC, transform, independent = gaussian(np.array(torques))
    trans2 = np.dot(trans, transform)
    return irred_elements[independent], trans2

def get_triplets_with_spg(pairs, symmetry, cell):
    first_atoms = symmetry.get_independent_atoms()
    symprec = symmetry.get_symmetry_tolerance()
    triplets = []

    positions = cell.get_scaled_positions()
    for i, atom1 in enumerate(first_atoms):
        site_sym = pairs[i]['site_symmetry']
        iatoms2 = pairs[i]['independent_atoms']
        pairs[i]['next_atoms'] = []
        for j, atom2 in enumerate(iatoms2):
            reduced_bond_sym = get_bond_symmetry(
                site_sym,
                positions,
                atom1,
                atom2,
                symprec)
            second_atom = get_next_atom(atom2, reduced_bond_sym, positions, symprec)
            pairs[i]['next_atoms'].append(second_atom)
            for k, atom3 in enumerate(second_atom['independent_atoms']):
                triplets.append([[atom1, atom2, atom3]])
    return triplets

def get_tripelts_with_permute(pairs, symmetry, cell):
    positions = cell.get_scaled_positions()
    symprec = symmetry.get_symmetry_tolerance()
    triplets_reduced = []
    for i1, p1 in enumerate(pairs):
        for i2, p2 in enumerate(p1['next_atoms']):
            for i3, p3 in enumerate(p2['next_atoms']):
                triplet = (p1['atom_number'], p2['atom_number'], p3['atom_number'])
                index = (i1, i2, i3)
                is_exist = False
                for permute in list(permutations([0,1,2]))[1:]: # Leave out the original triplet
                    atom1, atom2, atom3 = [triplet[i] for i in permute] # permuted triplet
                    if triplet == (atom1, atom2, atom3):
                        is_exist = False
                        continue
                        # atom1 = triplet[permute_matrix[0]];atom2 = triplet[permute_matrix[1]]; atom3 = triplet[permute_matrix[2]] # permuted triplet
                    numope = symmetry.get_map_operations()[atom1] #number of symmetry operation
                    rot, tran = (symmetry.get_symmetry_operation(numope)[ope] for ope in ("rotations", "translations"))
                    atom1 = symmetry.get_map_atoms()[atom1]
                    atom2 = get_atom_sent_by_operation(atom2, positions, rot, tran, symprec=symprec)
                    atom3 = get_atom_sent_by_operation(atom3, positions, rot, tran, symprec=symprec)
                    # atom1 is transformed to an independent atom and atom2 and atom3 are transformed following atom1
                    index1 = np.where(symmetry.get_independent_atoms() == atom1)[0][0] # index of the first irreducible atom
                    site_sym = pairs[index1]['site_symmetry']
                    mapping2 = pairs[index1]['mapping'] # now atom1 would be fixed under its site symmetries
                    iatoms2 = pairs[index1]['independent_atoms']
                    rot2 = site_sym[pairs[index1]['mapping_operation'][atom2]]
                    # a rotation which transforms atom2 to an independent one under site symmetries
                    atom2 = mapping2[atom2]
                    atom3 = get_atom_sent_by_site_sym(atom3, atom1, positions, rot2, symprec=symprec)
                    # atom3 is transformed following atom2
                    index2 = np.where(iatoms2 == atom2)[0][0]# index of the second irreducible atom
                    site_sym3 = pairs[index1]['next_atoms'][index2]['site_symmetry']
                    map_ope3 = pairs[index1]['next_atoms'][index2]['mapping_operation']
                    mapping3 = pairs[index1]['next_atoms'][index2]['mapping']
                    rot3 = site_sym3[map_ope3[atom3]]
                    atom3 = mapping3[atom3] # atom3 is also transformed to an irreducible one under bond symmetries
                    index3 = np.where(pairs[index1]['next_atoms'][index2]['independent_atoms'] == atom3)[0][0]

                    triplet_permute = [atom1, atom2, atom3]
                    # Check if triplet_permute already exists in the triplets pool
                    is_found=False
                    for tr in triplets_reduced:
                        if np.all(np.subtract(tr, triplet_permute)==0):
                            is_found = True
                            break
                    if is_found:
                        is_exist = True
                        tunnel = {}
                        tunnel["star"] = (index1, index2, index3)
                        # tunnel['transformation'] = np.dot(rot3, np.dot(rot2, np.dot(rot, permute_matrix)))
                        transf = np.dot(rot3, np.dot(rot2, rot))
                        transf_cart = similarity_transformation(cell.get_cell().T, transf)
                        seq = "".join(["ikm"[i] for i in permute])
                        tunnel['transformation'] = np.einsum("ij,kl,mn -> %sjln"%seq,
                                                             transf_cart, transf_cart, transf_cart).reshape(27,27)
                        ##for testing
                        # if (np.abs(np.dot(tunnel['transformation'], fc3[triplet[0], triplet[1], triplet[2]].flatten()) -
                        #         fc3[atom1, atom2, atom3].flatten()) > 1e-3).any():
                        #     print triplet
                        # the transformation is a matrix that further transfroms fc3[atom1, atom2, atom3] into its irreducible one
                        # by considering permutation symmetries
                        # permutation on fc3 is the same as swapping axes
                        pairs[index[0]]['next_atoms'][index[1]]['next_atoms'][index[2]]['tunnel'] = tunnel
                        break
                if not is_exist:
                    triplets_reduced.append(triplet)
                    pairs[index[0]]['next_atoms'][index[1]]['next_atoms'][index[2]]['tunnel'] = None
    return triplets_reduced

def get_fc3_coefficient_and_mapping(symmetry, cell, pairs, triplets):
    natom = cell.get_number_of_atoms()
    positions = cell.get_scaled_positions()
    symprec = symmetry.get_symmetry_tolerance()
    ifc_map = np.zeros((natom, natom, natom), dtype=np.int16)
    coeff = np.zeros((natom, natom, natom, 27, 27), dtype=np.float)

    for atom1 in np.arange(natom):
        iratom1 = symmetry.get_map_atoms()[atom1]
        numope = symmetry.get_map_operations()[atom1]
        index1 = np.where(symmetry.get_independent_atoms() == iratom1)[0][0]
        rot1, tran1 = (symmetry.get_symmetry_operation(numope)[ope] for ope in ("rotations", "translations"))
        p2 = pairs[index1]
        for atom2 in np.arange(natom):
            iratom2 = get_atom_sent_by_operation(atom2, positions, rot1, tran1, symprec=symprec)
            rot2 = p2['site_symmetry'][p2['mapping_operation'][iratom2]]
            iratom2 = p2['mapping'][iratom2]
            index2 = np.where(p2['independent_atoms'] == iratom2)[0][0]
            p3 = p2['next_atoms'][index2]
            for atom3 in np.arange(natom):
                iratom3 = get_atom_sent_by_operation(atom3, positions, rot1, tran1, symprec=symprec)
                iratom3 = get_atom_sent_by_site_sym(iratom3, iratom1, positions, rot2, symprec=symprec)
                rot3 = p3['site_symmetry'][p3['mapping_operation'][iratom3]]
                iratom3 = p3['mapping'][iratom3]
                index3 = np.where(p3['independent_atoms'] == iratom3)[0][0]
                rot = np.dot(rot3, np.dot(rot2, rot1))
                #rot transforms an arbitrary triplet to the irreducible one
                star3 = p3['next_atoms'][index3]
                rot_cart = np.double(similarity_transformation(cell.get_cell().T, rot))
                coeff_temp = np.einsum("ij, kl, mn -> ikmjln", rot_cart, rot_cart, rot_cart).reshape(27,27)
                if star3['tunnel'] is None:
                    ifc_map[atom1, atom2, atom3] = triplets.index((iratom1, iratom2, iratom3))
                else:
                    i1, i2, i3 = star3['tunnel']['star']
                    a1 = pairs[i1]['atom_number']
                    a2 = pairs[i1]['next_atoms'][i2]['atom_number']
                    a3 = pairs[i1]['next_atoms'][i2]['next_atoms'][i3]['atom_number']
                    permu_trans = star3['tunnel']['transformation']
                    ifc_map[atom1, atom2, atom3] = triplets.index((a1, a2, a3))
                    coeff_temp = np.dot(permu_trans, coeff_temp)
                    #take the permutation matrix into consideration
                coeff[atom1, atom2, atom3] = coeff_temp.T # inverse equals transpose
                # m1, m2, m3 = triplets_reduced[ifc_map[atom1, atom2, atom3]]
                # if not (np.abs(np.dot(coeff[atom1, atom2, atom3], fc3[m1, m2, m3].flatten()) -
                #     fc3[atom1, atom2, atom3].flatten()) < 1e-3).all():
                #     print atom1, atom2, atom3
    return  coeff, ifc_map


def set_symmetry_invariance(symmetry, cell, pairs,triplets_reduced):
    positions = cell.get_scaled_positions()
    symprec = symmetry.get_symmetry_tolerance()
    triplets_dict = []
    for triplet in triplets_reduced:
        triplet_dict = {}
        triplet_dict["triplet"] = triplet
        CC = [np.zeros(27)]

        for permute in list(permutations([0,1,2])): # Leave out the original triplet
            rot_all = []
            atom1, atom2, atom3 = [triplet[i] for i in permute]
            if not symmetry.get_map_atoms()[atom1] == triplet[0]:
                continue
            for numope in get_operations_at_star(symmetry.get_symmetry_operations(),
                                                 positions,
                                                 atom1,
                                                 symmetry.get_map_atoms(),
                                                 symprec):
                #get all the symmetry operation that keeps atom1 unchanged
                rot1, tran = (symmetry.get_symmetry_operation(numope)[ope] for ope in ("rotations", "translations"))
                atom1_1 = symmetry.get_map_atoms()[atom1]
                index1 = np.where(symmetry.get_independent_atoms() == atom1_1)[0][0] # index of the first irreducible atom
                atom2_1 = get_atom_sent_by_operation(atom2, positions, rot1, tran, symprec=symprec)
                atom3_1 = get_atom_sent_by_operation(atom3, positions, rot1, tran, symprec=symprec)
                site_syms = pairs[index1]['site_symmetry']
                mapping2 = pairs[index1]['mapping'] # now atom1 would be fixed under its site symmetries
                iatoms2 = pairs[index1]['independent_atoms']
                if mapping2[atom2_1] != triplet[1]:
                    continue
                for rot2 in get_rotations_at_star(site_syms, positions, atom1_1, atom2_1, mapping2, symprec):
                    atom2_2 = mapping2[atom2_1]
                    atom3_2 = get_atom_sent_by_site_sym(atom3_1, atom1_1, positions, rot2, symprec=symprec) #no matter atom1_1 or atom2_1
                    index2 = np.where(iatoms2 == atom2_2)[0][0]# index of the second irreducible atom
                    site_syms3 = pairs[index1]['next_atoms'][index2]['site_symmetry']
                    mapping3 = pairs[index1]['next_atoms'][index2]['mapping']
                    if not mapping3[atom3_2] == triplet[2]:
                        continue
                    for rot3 in get_rotations_at_star(site_syms3, positions, atom2_2, atom3_2, mapping3, symprec):
                        rot = np.dot(rot3, np.dot(rot2, rot1))
                        isfound = False
                        for r in rot_all:
                            if (np.abs(r-rot) < 1e-8).all():
                                isfound = True
                                break
                        if not isfound:
                            rot_all.append(rot)
                        else:
                            continue
                        rot_cart = np.double(similarity_transformation(cell.get_cell().T, rot))
                        seq = "".join(["ikm"[i] for i in permute])
                        PP  = np.einsum("ij,kl,mn -> %sjln"%seq, rot_cart, rot_cart, rot_cart).reshape(27, 27).T
                        BB = PP - np.eye(27)
                        for i in np.arange(27):
                            is_found = False
                            if not (np.abs(BB[i]) < 1e-8).all():
                                for j in np.arange(len(CC)):
                                    if (np.abs(BB[i] - CC[j]) < 1e-8).all():
                                        is_found = True
                                        break
                                if not is_found:
                                    CC.append(BB[i])
        CC, transform, independent = gaussian(np.array(CC))
        # if (np.abs(np.dot(transform, fc3[triplet[0], triplet[1], triplet[2]].flatten()[independent]) -
        #         fc3[triplet[0], triplet[1], triplet[2]].flatten()) > 1e-3).any():
        #     print triplet
        triplet_dict['independent'] = independent
        triplet_dict['transform'] = transform
        triplets_dict.append(triplet_dict)
    num_irred = [len(dic['independent']) for dic in triplets_dict]
    trans = np.zeros((len(triplets_dict), 27, sum(num_irred)), dtype="float")
    for i, trip  in enumerate(triplets_dict):
        start = sum(num_irred[:i])
        length = num_irred[i]
        trans[i,:, start:start + length] = trip['transform']
    return trans

def set_acoustic_summation(trans, coeff, ifc_map, cell, symmetry, pairs):
    # set translational invariance
    first_atoms = symmetry.get_independent_atoms()
    num_irred = trans.shape[-1]
    natom = cell.get_number_of_atoms()
    ti_transforms =[]
    for i, atom1 in enumerate(first_atoms):
        iatoms2 = pairs[i]['independent_atoms']
        for j, atom2 in enumerate(iatoms2):
            ti_transform = np.zeros((27, num_irred))
            for atom3 in range(natom):
                irred_triplet = ifc_map[atom1, atom2, atom3]
                transform = trans[irred_triplet]
                ti_transform += np.dot(coeff[atom1, atom2, atom3], transform)
            for k in range(27):
                if not (np.abs(ti_transform[k])< 1e-8).all():
                    ti_transforms.append(ti_transform[k])
    CC, transform, independent = gaussian(np.array(ti_transforms))
    return np.dot(trans, transform)

def set_rotational_invariance(fc2, trans, coeff, ifc_map, cell, symmetry, pairs):
    natom = cell.get_number_of_atoms()
    num_irred2 = trans.shape[-1]
    eijk = np.zeros((3,3,3), dtype="intc")
    eijk[0,1,2] = eijk[1,2,0] = eijk[2,0,1] = 1
    eijk[0,2,1] = eijk[2,1,0] = eijk[1,0,2] = -1 # epsion matrix, which is an antisymmetric 3 * 3 * 3 tensor
    torques =[np.zeros(num_irred2)]
    lattice = cell.get_cell()
    for i, atom1 in enumerate(symmetry.get_independent_atoms()):
        iatoms2 = pairs[i]['independent_atoms']
        for j, atom2 in enumerate(iatoms2):
            torque = np.zeros((27, num_irred2), dtype=np.float)
            for atom3 in range(natom):
                fc_temp = np.dot(coeff[atom1, atom2, atom3], trans[ifc_map[atom1, atom2, atom3]]).reshape(3,3,3, -1)
                vectors = get_equivalent_smallest_vectors(atom3,
                                                          atom1,
                                                          cell,
                                                          lattice,
                                                          symmetry.get_symmetry_tolerance())
                r_frac = np.array(vectors).sum(axis=0) / len(vectors)
                r = np.dot(r_frac, lattice)
                disp = np.einsum("j, ijk -> ik", r, eijk)
                t3 = np.einsum("abcN, cv -> abvN", fc_temp, disp).reshape(27, num_irred2)
                t2_1 = np.einsum("bc, cav -> abv", fc2[atom1, atom2], eijk).flatten()
                t2_2 = np.einsum("ac, cbv -> abv", fc2[atom1, atom2], eijk).flatten()
                torque += (t3.T + t2_1 + t2_2).T
            for k in range(27):
                if not (np.abs(torque[k])< 1e-5).all():
                    torques.append(torque[k])
    CC, transform, independent = gaussian(np.array(torques))
    trans2 = np.dot(trans, transform)
    return trans2

def get_fc3_irreducible_components(cell,symmetry, fc2):
    pairs = get_pairs(symmetry, cell)
    get_triplets_with_spg(pairs, symmetry, cell)
    triplets_reduced = get_tripelts_with_permute(pairs, symmetry, cell)
    trans = set_symmetry_invariance(symmetry, cell, pairs,triplets_reduced)
    coeff, ifc_map = get_fc3_coefficient_and_mapping(symmetry, cell, pairs, triplets_reduced)
    # trans = set_symmetry_invariance(symmetry, cell, pairs,triplets_reduced)
    trans = set_acoustic_summation(trans, coeff, ifc_map, cell, symmetry, pairs)
    trans = set_rotational_invariance(fc2, trans, coeff, ifc_map, cell, symmetry, pairs)
    return coeff, ifc_map, trans

def show_drift_fc3(fc3, name="fc3"):
    num_atom = fc3.shape[0]
    maxval1 = 0
    maxval2 = 0
    maxval3 = 0
    for i, j, k, l, m in list(np.ndindex((num_atom, num_atom, 3, 3, 3))):
        val1 = fc3[:, i, j, k, l, m].sum()
        val2 = fc3[i, :, j, k, l, m].sum()
        val3 = fc3[i, j, :, k, l, m].sum()
        if abs(val1) > abs(maxval1):
            maxval1 = val1
        if abs(val2) > abs(maxval2):
            maxval2 = val2
        if abs(val3) > abs(maxval3):
            maxval3 = val3
    print ("max drift of %s:" % name), maxval1, maxval2, maxval3