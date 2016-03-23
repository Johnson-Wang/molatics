__author__ = 'xinjiang'
import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation
from realmd.information import timeit
from itertools import permutations
from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors

@timeit
def get_irreducible_components2(symmetry, supercell):
    second_independent_atom = []
    second_mapping_operations = []
    second_mapping_atoms = []
    positions = supercell.get_scaled_positions()
    lattice = supercell.get_cell()
    symprec = symmetry.get_symmetry_tolerance()
    pairs = []
    for pos_p in symmetry.get_independent_atoms():
        pair = {}
        pair['atom_number'] = pos_p
        rela_pos = positions - positions[pos_p]
        map_atoms = range(supercell.get_number_of_atoms())
        site_symmetry = symmetry.get_site_symmetry(pos_p)
        pair['site_symmetry'] = site_symmetry
        for i, p in enumerate(rela_pos):
            is_found = False
            for j in range(i):
                for r in site_symmetry:
                    diff = np.dot(p, r.T) - rela_pos[j]
                    diff -= np.rint(diff)
                    dist = np.linalg.norm(np.dot(diff, lattice))
                    if dist < symprec:
                        map_atoms[i] = j
                        is_found = True
                        break
                if is_found:
                    break
        map_atoms = np.array(map_atoms, dtype=int)
        pair['mapping'] = map_atoms
        map_ops = np.zeros(len(rela_pos), dtype=int)
        independent_atoms = []
        for i, eq_atom in enumerate(map_atoms):
            if i == eq_atom:
                independent_atoms.append(i)
            for j, r in enumerate(site_symmetry):
                diff = np.dot(rela_pos[i], r.T) - rela_pos[eq_atom]
                if (abs(diff - np.rint(diff)) < symprec).all():
                    map_ops[i] = j
                    break
        pair['mapping_operation'] = map_ops
        second_mapping_atoms.append(map_atoms)
        second_mapping_operations.append(map_ops)
        second_independent_atom.append(independent_atoms)
        pairs.append(pair)
    return second_independent_atom, second_mapping_atoms, second_mapping_operations

def get_next_atom(center, site_symmetry, positions, symprec=1e-5):
    """next_atom: a dict which should at least contain the key 'atom_number'
    site_symmetry: the site_symmetry at the center atom (atom_number)"""
    next_atom = {}
    next_atom['atom_number'] = center
    rela_pos = positions - positions[center]
    map_atoms = np.arange(len(positions))
    map_ops = np.zeros(len(positions), dtype=int)
    next_atom['site_symmetry'] = site_symmetry
    for i, p in enumerate(rela_pos):
        is_found = False
        for j in range(i):
            for k,r in enumerate(site_symmetry):
                diff = np.dot(p, r.T) - rela_pos[j]
                diff -= np.rint(diff)
                if (abs(diff) < symprec).all():
                    map_atoms[i] = j
                    map_ops[i] = k
                    is_found = True
                    break
            if is_found:
                break
    next_atom['mapping'] =map_atoms
    next_atom['mapping_operation'] = map_ops
    next_atom['independent_atoms'] = np.unique(map_atoms)
    next_atom['next_atoms'] = [{"atom_number":a} for a in np.unique(map_atoms)]
    return next_atom


def get_atom_mapping(center, site_symmetry, positions, symprec=1e-5, is_return_opes=False):
    """next_atom: a dict which should at least contain the key 'atom_number'
    site_symmetry: the site_symmetry at the center atom (atom_number)"""
    next_atom = {}
    rela_pos = positions - positions[center]
    map_atoms = np.arange(len(positions))
    map_ops = np.zeros(len(positions), dtype=int)
    for i, p in enumerate(rela_pos):
        is_found = False
        for j in range(i):
            for k,r in enumerate(site_symmetry):
                diff = np.dot(p, r.T) - rela_pos[j]
                diff -= np.rint(diff)
                if (abs(diff) < symprec).all():
                    map_atoms[i] = j
                    map_ops[i] = k
                    is_found = True
                    break
            if is_found:
                break
    if is_return_opes:
        return map_atoms, map_ops
    else:
        return map_atoms

def get_pairs(symmetry, supercell):
    positions = supercell.get_scaled_positions()
    symprec = symmetry.get_symmetry_tolerance()
    pairs = []
    for pos_p in symmetry.get_independent_atoms():
        pair = get_next_atom(pos_p, symmetry.get_site_symmetry(pos_p), positions, symprec)
        pairs.append(pair)
    return pairs


def get_doublets_with_permute(pairs, symmetry, cell):
########set permutation symmetry################
    positions = cell.get_scaled_positions()
    doublets_reduced = []
    for i, p1 in enumerate(pairs):
        atom1 = p1['atom_number']
        for j, p2 in enumerate(p1['next_atoms']):
            atom2 = p2['atom_number']
            #Execute the permutation
            patom1, patom2 = atom2, atom1
            iratom1 = symmetry.get_map_atoms()[patom1]
            index1 = np.where(symmetry.get_independent_atoms() == iratom1)[0][0] # index of the first irreducible atom
            numope = symmetry.get_map_operations()[patom1] #number of symmetry operation
            rot, tran = (symmetry.get_symmetry_operation(numope)[ope] for ope in ("rotations", "translations"))
            iratom2 = get_atom_sent_by_operation(patom2, positions, rot, tran)
            rot2 = pairs[index1]['site_symmetry'][pairs[index1]['mapping_operation'][iratom2]]
            iratom2 = pairs[index1]['mapping'][iratom2]
            index2 = np.where(pairs[index1]['independent_atoms'] == iratom2)[0][0]# index of the second irreducible atom

            if not (iratom1, iratom2) in doublets_reduced:
                doublets_reduced.append((atom1, atom2))
                pairs[index1]['next_atoms'][index2]['tunnel'] = None
            else:
                if iratom1 == atom1 and iratom2 == atom2:
                    continue
                tunnel ={}
                tunnel['star'] = (index1, index2)
                transf = np.dot(rot2, rot)
                transf_cart = similarity_transformation(cell.get_cell().T, transf)
                tunnel['transformation'] = np.einsum("ij,km -> kijm", transf_cart, transf_cart).reshape(9,9)
                pairs[index1]['next_atoms'][index2]['tunnel'] = tunnel
    return doublets_reduced

def get_pairs_with_permute(pairs,
                           lattice,
                           positions,
                           rots1,
                           trans1,
                           ind_atoms1,
                           mappings1,
                           map_ope1,
                           rots2,
                           mappings2,
                           map_ope2):
########set permutation symmetry################
    pairs_mapping = np.arange(len(pairs))
    transform = np.zeros((len(pairs), 9, 9), dtype=np.float)
    for i, (atom1, atom2) in enumerate(pairs):
        #Execute the permutation
        patom1, patom2 = atom2, atom1
        iratom1 = mappings1[patom1]
        index1 = np.where(ind_atoms1 == iratom1)[0][0] # index of the first irreducible atom
        numope = map_ope1[patom1] #number of symmetry operation
        rot, tran = rots1[numope], trans1[numope]
        iratom2 = get_atom_sent_by_operation(patom2, positions, rot, tran)
        rot2 = rots2[index1][map_ope2[index1, iratom2]]
        iratom2 = mappings2[index1, iratom2]
        index_pair = pairs.index((iratom1, iratom2))
        if pairs_mapping[index_pair] < i:
            pairs_mapping[i] = pairs_mapping[index_pair]
            transf = np.dot(rot2, rot)
            transf_cart = similarity_transformation(lattice, transf)
            transf = np.einsum("ij,km -> kijm", transf_cart, transf_cart).reshape(9,9)
            transform[i] = np.dot(transform[index_pair], transf) # from a general to the irreducible one
        else:
            transform[i] = np.eye(9)
    return pairs_mapping, transform

def get_fc2_coefficients(pairs_orig,
                         mapping_pair,
                         transform_pair,
                         lattice,
                         positions,
                         rots1,
                         trans1,
                         ind_atoms1,
                         mappings1,
                         map_ope1,
                         rots2,
                         mappings2,
                         map_ope2):
    natom = len(positions)
    pairs_reduced = [pairs_orig[i] for i in np.unique(mapping_pair)]
    coeff = np.zeros((natom, natom, 9, 9), dtype=np.float)
    ifc_map = np.zeros((natom, natom), dtype="intc")
    for atom1 in np.arange(natom):
        r1, t = rots1[map_ope1[atom1]], trans1[map_ope1[atom1]]
        a1 = mappings1[atom1]
        index1 = np.where(ind_atoms1 == a1)[0][0] # index of the first irreducible atom
        site_symmetries = rots2[index1] #[:nope[index1]]
        for atom2 in np.arange(natom):
            map_atom2 = get_atom_sent_by_operation(atom2, positions, r1, t)
            a2 = mappings2[index1][map_atom2]
            r2 = site_symmetries[map_ope2[index1,map_atom2]]
            R = np.dot(r2, r1)
            R_cart = np.double(similarity_transformation(lattice, R))
            coeff_temp = np.einsum("ij, kl -> ikjl", R_cart, R_cart).reshape(9,9)
            index_orig = pairs_orig.index((a1, a2))
            star2 = pairs_orig[mapping_pair[index_orig]]
            ifc_map[atom1, atom2] = pairs_reduced.index(star2)
            permu_trans = transform_pair[index_orig]
            coeff_temp = np.dot(permu_trans, coeff_temp)
            #take the permutation matrix into consideration
            coeff[atom1, atom2] = coeff_temp.T # inverse equals transpose
    return coeff, ifc_map

def get_fc2_spg_invariance(pairs_orig,
                           is_pair_included,
                           positions,
                           rotations1,
                           translations1,
                           mappings1,
                           rotations2,
                           num_rotations2,
                           mappings2,
                           lattice,
                           symprec):
    indatoms1 = np.unique(mappings1)
    doublets_dict = []
    for index_pair,pair in enumerate(pairs_orig):
        if not is_pair_included[index_pair]:
            doublet_dict={"independent":[], "transform":None}
            doublets_dict.append(doublet_dict)
            continue
        doublet_dict = {}
        CC = [np.zeros(9)]
        for permute in list(permutations([0,1])):
            rot_all = []
            atom1, atom2= [pair[i] for i in permute]
            if not mappings1[atom1] == pair[0]:
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
                index1 = np.where(indatoms1 == atom1_1)[0][0] # index of the first irreducible atom
                atom2_1 = get_atom_sent_by_operation(atom2, positions, rot1, tran, symprec=symprec)
                site_syms = rotations2[index1][:num_rotations2[index1]]
                if mappings2[index1, atom2_1] != pair[1]:
                    continue
                for rot2 in get_rotations_at_star(site_syms, positions, atom1_1, atom2_1, mappings2[index1], symprec):
                    rot = np.dot(rot2, rot1)
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
                    seq = "".join(["ik"[i] for i in permute])
                    PP  = np.einsum("ij,kl -> %sjl"%seq, rot_cart, rot_cart).reshape(9, 9).T
                    BB = PP - np.eye(9)
                    for i in np.arange(9):
                        is_found = False
                        if not (np.abs(BB[i]) < 1e-8).all():
                            for j in np.arange(len(CC)):
                                if (np.abs(BB[i] - CC[j]) < 1e-8).all():
                                    is_found = True
                                    break
                            if not is_found:
                                CC.append(BB[i])
        CC, transform, independent = gaussian(np.array(CC))
        doublet_dict['independent'] = [ind + index_pair * 9 for ind in independent] # independent ele
        doublet_dict['transform'] = transform
        doublets_dict.append(doublet_dict)
    num_irred = [len(dic['independent']) for dic in doublets_dict]
    trans = np.zeros((len(doublets_dict), 9, sum(num_irred)), dtype="float")
    for i, doub  in enumerate(doublets_dict):
        start = sum(num_irred[:i])
        length = num_irred[i]
        trans[i,:, start:start + length] = doub['transform']
    return np.hstack((dic['independent'] for dic in doublets_dict)).astype("intc"), trans

def get_fc2_translational_invariance(trans2, irred_elements, coeff, ifc_map, indatoms1, positions):

    natom = len(positions)
    num_irred = len(irred_elements)
    ti_transforms =[]
    for i, atom1 in enumerate(indatoms1):
        ti_transform = np.zeros((9, num_irred))
        for atom2 in np.arange(natom):
            irred_doublet = ifc_map[atom1, atom2]
            ti_transform += np.dot(coeff[atom1, atom2], trans2[irred_doublet])
        for k in range(9):
            if not (np.abs(ti_transform[k])< 1e-8).all():
                ti_transforms.append(ti_transform[k])
    print "Number of constraints of fc2 from translational invariance:%d"%len(ti_transforms)

    CC, transform, independent = gaussian(np.array(ti_transforms))
    trans = np.dot(trans2, transform)
    return irred_elements[independent], trans


def get_fc2_rotational_invariance(cell, primitive, trans, irred_elements, coeff, ifc_map, indatoms1, symprec):
    positions = cell.get_scaled_positions()
    natom = len(positions)
    num_irred2 = len(irred_elements)
    lattice = primitive.get_cell()
    eijk = np.zeros((3,3,3), dtype="intc")
    eijk[0,1,2] = eijk[1,2,0] = eijk[2,0,1] = 1
    eijk[0,2,1] = eijk[2,1,0] = eijk[1,0,2] = -1 # epsion matrix, which is an antisymmetric 3 * 3 * 3 tensor
    torques =[np.zeros(num_irred2)]
    for i, patom_num in enumerate(indatoms1):
        torque = np.zeros((9, num_irred2), dtype=np.float)
        for j in range(natom):
            fc_temp = np.dot(coeff[patom_num, j], trans[ifc_map[patom_num, j]]).reshape(3,3,-1)
            vectors = get_equivalent_smallest_vectors(j,
                                                      patom_num,
                                                      cell,
                                                      lattice,
                                                      symprec)
            r_frac = np.array(vectors).sum(axis=0) / len(vectors)
            r = np.dot(r_frac, lattice)
            disp = np.einsum("j, ijk -> ik", r, eijk)
            torque += np.einsum("abN, bv -> avN", fc_temp, disp).reshape(9, num_irred2)
        for k in range(9):
            if not (np.abs(torque[k])< 1e-8).all():
                torques.append(torque[k])
    import h5py
    f = h5py.File("fc2.hdf5")
    fc2 = f['fc2'][:]
    f.close()
    fc2_pairs = fc2.reshape(-1, 3, 3)[np.unique(ifc_map.flatten(), return_index=True)[1]]
    fc2_reduced = fc2_pairs.flatten()[irred_elements]
    print "Number of constraints of fc2 from rotational invariance:%d"%len(torques)
    CC, transform, independent = gaussian(np.array(torques))
    trans = np.dot(trans, transform)
    return irred_elements[independent], trans

def get_fc2_coefficient_and_mapping(doublets_reduced, pairs, symmetry, cell):
    natom = cell.get_number_of_atoms()
    positions = cell.get_scaled_positions()
    coeff = np.zeros((natom, natom, 9, 9), dtype=np.float)
    ifc_map = np.zeros((natom, natom), dtype="intc")
    for atom1 in np.arange(natom):
        nmap = symmetry.get_map_operations()[atom1]
        map_syms = symmetry.get_symmetry_operation(nmap)
        a1 = symmetry.get_map_atoms()[atom1]
        index1 = np.where(symmetry.get_independent_atoms() == a1)[0][0] # index of the first irreducible atom
        site_symmetries = pairs[index1]['site_symmetry']
        r1 = map_syms['rotations']
        t = map_syms['translations']
        for atom2 in np.arange(natom):
            map_atom2 = get_atom_sent_by_operation(atom2, positions, r1, t)
            a2 = pairs[index1]['mapping'][map_atom2]
            index2 = np.where(pairs[index1]['independent_atoms'] == a2)[0][0]# index of the second irreducible atom
            r2 = site_symmetries[pairs[index1]['mapping_operation'][map_atom2]]
            R = np.dot(r2, r1)
            R_cart = np.double(similarity_transformation(cell.get_cell().T, R))
            coeff_temp = np.einsum("ij, kl -> ikjl", R_cart, R_cart).reshape(9,9)
            star2 = pairs[index1]['next_atoms'][index2]
            if star2['tunnel'] is None:
                ifc_map[atom1, atom2] = doublets_reduced.index((a1, a2))
            else:
                i1, i2 = star2['tunnel']['star']
                a1 = pairs[i1]['atom_number']
                a2 = pairs[i1]['next_atoms'][i2]['atom_number']
                permu_trans = star2['tunnel']['transformation']
                ifc_map[atom1, atom2] = doublets_reduced.index((a1, a2))
                coeff_temp = np.dot(permu_trans, coeff_temp)
                #take the permutation matrix into consideration
            coeff[atom1, atom2] = coeff_temp.T # inverse equals transpose
    return coeff, ifc_map


def set_spg_invariance(doublets_reduced, pairs, symmetry, cell):
    positions = cell.get_scaled_positions()
    symprec = symmetry.get_symmetry_tolerance()
    doublets_dict = []
    for doublet in doublets_reduced:
        doublet_dict = {}
        doublet_dict["double"] = doublet
        CC = [np.zeros(9)]
        for permute in list(permutations([0,1])):
            rot_all = []
            atom1, atom2= [doublet[i] for i in permute]
            if not symmetry.get_map_atoms()[atom1] == doublet[0]:
                continue
            for numope in get_operations_at_star(symmetry.get_symmetry_operations(),
                                                 positions,
                                                 atom1,
                                                 symmetry.get_map_atoms(),
                                                 symmetry.get_symmetry_tolerance()):
                #get all the symmetry operation that keeps atom1 unchanged
                rot1, tran = (symmetry.get_symmetry_operation(numope)[ope] for ope in ("rotations", "translations"))
                atom1_1 = symmetry.get_map_atoms()[atom1]
                index1 = np.where(symmetry.get_independent_atoms() == atom1_1)[0][0] # index of the first irreducible atom
                atom2_1 = get_atom_sent_by_operation(atom2, positions, rot1, tran, symprec=symprec)
                site_syms = pairs[index1]['site_symmetry']
                mapping2 = pairs[index1]['mapping'] # now atom1 would be fixed under its site symmetries
                if mapping2[atom2_1] != doublet[1]:
                    continue
                for rot2 in get_rotations_at_star(site_syms, positions, atom1_1, atom2_1, mapping2, symprec):
                    rot = np.dot(rot2, rot1)
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
                    seq = "".join(["ik"[i] for i in permute])
                    PP  = np.einsum("ij,kl -> %sjl"%seq, rot_cart, rot_cart).reshape(9, 9).T
                    BB = PP - np.eye(9)
                    for i in np.arange(9):
                        is_found = False
                        if not (np.abs(BB[i]) < 1e-8).all():
                            for j in np.arange(len(CC)):
                                if (np.abs(BB[i] - CC[j]) < 1e-8).all():
                                    is_found = True
                                    break
                            if not is_found:
                                CC.append(BB[i])
        CC, transform, independent = gaussian(np.array(CC))
        doublet_dict['independent'] = independent
        doublet_dict['transform'] = transform
        doublets_dict.append(doublet_dict)
    num_irred = [len(dic['independent']) for dic in doublets_dict]
    trans = np.zeros((len(doublets_dict), 9, sum(num_irred)), dtype="float")
    for i, doub  in enumerate(doublets_dict):
        start = sum(num_irred[:i])
        length = num_irred[i]
        trans[i,:, start:start + length] = doub['transform']
    return trans

def set_acoustic_summation(trans2, coeff, ifc_map, cell, symmetry):
    natom = cell.get_number_of_atoms()
    num_irred = trans2.shape[-1]
    ti_transforms =[]
    for i, atom1 in enumerate(symmetry.get_independent_atoms()):
        ti_transform = np.zeros((9, num_irred))
        for atom2 in np.arange(natom):
            irred_doublet = ifc_map[atom1, atom2]
            ti_transform += np.dot(coeff[atom1, atom2], trans2[irred_doublet])
        for k in range(9):
            if not (np.abs(ti_transform[k])< 1e-8).all():
                ti_transforms.append(ti_transform[k])
    CC, transform, independent = gaussian(np.array(ti_transforms))
    trans = np.dot(trans2, transform)
    return trans

def set_rotation_inv(trans, coeff, ifc_map, cell, symmetry):
    natom = cell.get_number_of_atoms()
    num_irred2 = trans.shape[-1]
    eijk = np.zeros((3,3,3), dtype="intc")
    eijk[0,1,2] = eijk[1,2,0] = eijk[2,0,1] = 1
    eijk[0,2,1] = eijk[2,1,0] = eijk[1,0,2] = -1 # epsion matrix, which is an antisymmetric 3 * 3 * 3 tensor
    torques =[np.zeros(num_irred2)]
    lattice = cell.get_cell()
    for i, patom_num in enumerate(symmetry.get_independent_atoms()):
        torque = np.zeros((9, num_irred2), dtype=np.float)
        for j in range(natom):
            fc_temp = np.dot(coeff[patom_num, j], trans[ifc_map[patom_num, j]]).reshape(3,3,-1)
            vectors = get_equivalent_smallest_vectors(j,
                                                      patom_num,
                                                      cell,
                                                      lattice,
                                                      symmetry.get_symmetry_tolerance())
            r_frac = np.array(vectors).sum(axis=0) / len(vectors)
            r = np.dot(r_frac, lattice)
            disp = np.einsum("j, ijk -> ik", r, eijk)
            torque += np.einsum("abN, bv -> avN", fc_temp, disp).reshape(9, num_irred2)
        for k in range(9):
            if not (np.abs(torque[k])< 1e-8).all():
                torques.append(torque[k])
    CC, transform, independent = gaussian(np.array(torques))
    trans2 = np.dot(trans, transform)
    return trans2

def get_fc2_least_irreducible_components(symmetry,
                                         supercell,
                                         is_rot_inv=False,
                                         is_trans_inv=False,
                                         cutoff_pair=None,
                                         log_level=0):
    if log_level:
        print ""
    pairs = get_pairs(symmetry, supercell)
    doublets_reduced = get_doublets_with_permute(pairs, symmetry, supercell)

    coeff, ifc_map = get_fc2_coefficient_and_mapping(doublets_reduced, pairs, symmetry, supercell)
    trans = set_spg_invariance(doublets_reduced, pairs, symmetry, supercell)
    if is_trans_inv and cutoff_pair == None:
        trans = set_acoustic_summation(trans, coeff, ifc_map, supercell, symmetry)
    if is_rot_inv and cutoff_pair == None:
        trans = set_rotation_inv(trans, coeff, ifc_map, supercell, symmetry)
    return coeff, ifc_map, trans


def get_disp_coefficient(symmetry, lattice, positions, irred_atoms2, map_atoms2, map_opes2):
    # import h5py; fc2 = h5py.File("fc2.hdf5")['fc2'][:]
    natom = len(positions)
    symprec = symmetry.get_symmetry_tolerance()
    # num_ifc = len(sum([i[1] for i in ifc], []))
    len2 = np.cumsum([0] + [len(i) for i in irred_atoms2]) # cumulative summation of the second irreducible atoms
    irred_atoms1 = symmetry.get_independent_atoms()
    map_ope1 = symmetry.get_map_operations()
    coeff = np.zeros((natom, natom,3,3,3,3), dtype="intc")
    ifc_map = np.zeros((natom, natom), dtype="intc")
    rotations = symmetry.get_symmetry_operations()['rotations']
    translations = symmetry.get_symmetry_operations()['translations']
    for atom1 in np.arange(natom):
        map_syms = map_ope1[atom1]
        map_atom = symmetry.get_map_atoms()[atom1]
        site_symmetries = symmetry.get_site_symmetry(map_atom)
        irr_index = np.where(irred_atoms1 == map_atom)[0][0]
        r1 = rotations[map_syms]
        t = translations[map_syms]
        for atom2 in np.arange(natom):
            rot_pos = np.dot(positions[atom2], r1.T) + t
            diff = positions - rot_pos
            map_atom2 = np.where(np.all(np.abs(diff - np.rint(diff)) < symprec, axis=1))[0][0]
            index = np.where(irred_atoms2[irr_index] == map_atoms2[irr_index][map_atom2])[0][0]
            ifc_map[atom1, atom2] = len2[irr_index] + index
            r2 = site_symmetries[map_opes2[irr_index][map_atom2]]
            # r2 transforms atom2 into its irreducible one
            # R transfroms an irreducible triplets to an arbitray one (note that r1.T equals rt^-1)
            R = np.dot(r1.T, r2.T)
            R_cart = np.double(similarity_transformation(lattice, R))
            # coeff[atom1, atom2] = np.einsum("ij, kl -> iljk", R_cart, np.linalg.inv(R_cart))
            coeff[atom1, atom2] = np.einsum("ij, kl -> ikjl", R_cart, R_cart)
            # assert np.all(np.dot(coeff[atom1, atom2].reshape(9,9), fc2[map_atom, map_atoms2[irr_index][map_atom2]].flatten()) -
            #         fc2[atom1, atom2].flatten() < 1e-5)

    return coeff, ifc_map


def get_atom_sent_by_operation(orig_atom, positions, r, t, symprec=1e-5):
    rot_pos = np.dot(positions[orig_atom], r.T) + t
    diff = positions - rot_pos
    atom2 = np.where(np.all(np.abs(diff - np.rint(diff)) < symprec, axis=1))[0][0]
    return atom2


def get_atom_sent_by_site_sym(atom, center, positions, rot, symprec=1e-5):
    rot_pos = np.dot(positions[atom] - positions[center], rot.T) + positions[center]
    diff = positions - rot_pos
    atom2 = np.where(np.all(np.abs(diff - np.rint(diff)) < symprec, axis=1))[0][0]
    return atom2


def get_operations_at_star(operations, positions, atom, mappings, symprec):
    star = mappings[atom]
    map_operations = []
    for j, (r, t) in enumerate(
            zip(operations['rotations'], operations['translations'])):
        diff = np.dot(positions[atom], r.T) + t - positions[star]
        if (abs(diff - np.rint(diff)) < symprec).all():
            map_operations.append(j)
    return map_operations

def get_all_operations_at_star(rotations, translations, positions, atom, mappings, symprec):
    star = mappings[atom]
    map_operations = []
    for j, (r, t) in enumerate(
            zip(rotations, translations)):
        diff = np.dot(positions[atom], r.T) + t - positions[star]
        if (abs(diff - np.rint(diff)) < symprec).all():
            map_operations.append(j)
    return map_operations

def get_rotations_at_star(site_symmetries, positions, center_atom, atom, mappings, symprec):
    star = mappings[atom]
    map_operations = []
    rel_pos = positions - positions[center_atom]
    for j, r in enumerate(site_symmetries):
        diff = np.dot(rel_pos[atom], r.T) - rel_pos[star]
        if (abs(diff - np.rint(diff)) < symprec).all():
            map_operations.append(j)
    return site_symmetries[map_operations]


def gaussian2(a, prec=1e-8):
    """Compute the gaussian elimination with row pivoting
    Parametere
    -------------------------------------------------
    a: (M, N) arraay_like
    array to decompose
    prec: float
    absolute values lower than prec are ignored
    Return:
    a: ((max(M, N), N) array_like
      the gaussian eliminated array
    param: (N, num_independent) array_like
      parameters which transfrom from the independent elements to all
    independent: (num_independent,) list
      indexes of independent elements"""
    dependent = []
    independent = []
    row, column = a.shape
    if row < column:
        a = np.vstack((a, np.zeros((column-row, column))))
    row, column = a.shape
    a = np.where(np.abs(a)<prec, 0, a).astype(np.float)
    irow = 0
    for i in range(column):
        for j in range(irow+1, row):
            if abs(a[j,i]) -abs(a[irow, i]) > prec:
                a[[irow, j]] = a[[j,irow]] # interchange the irowth and jth row

        if abs(a[irow, i]) > prec:
            dependent.append(i)
            a[irow, i:] /= a[irow, i]
            for j in range(irow) + range(irow+1, row):
                a[j, i:] -= a[irow, i:] / a[irow, i] * a[j,i]
            irow += 1
        else:
            independent.append(i)
    param = np.zeros((column, len(independent)), dtype=np.float)
    if len(independent) >0:
        for i, de in enumerate(dependent):
            for j, ind in enumerate(independent):
                param[de, j] = -a[i, ind]
        for i, ind in enumerate(independent):
            param[ind, i] = 1
    return a, param, independent


def gaussian(a, prec=1e-4):
    """Compute the gaussian elimination with both row and column pivoting
    Parametere
    -------------------------------------------------
    a: (M, N) array_like
    array to decompose
    prec: float
    absolute values lower than prec are ignored
    Return:
    a: ((max(M, N), N) array_like
      the gaussian eliminated array
    param: (N, num_independent) array_like
      parameters which transfrom from the independent elements to all
    independent: (num_independent,) list
      indexes of independent elements"""
    dependent = []
    independent = []
    # row, column = a.shape
    # if row < column:
    #     a = np.vstack((a, np.zeros((column-row, column))))
    row, column = a.shape
    a = np.where(np.abs(a)<prec, 0, a).astype(np.float)
    irow = 0
    for c in range(column):
        max_column = np.max(np.abs(a), axis=0)
        indices = [m for m in range(column) if m not in dependent+independent]
        i = indices[np.argmax(max_column[indices])] # find the column of the maximum element
        for j in range(irow+1, row):
            if abs(a[j,i]) -abs(a[irow, i]) > prec:
                a[[irow, j]] = a[[j,irow]] # interchange the irowth and jth row

        if abs(a[irow, i]) > prec:
            if column > 1000:
                print a[irow, i]
            dependent.append(i)
            a[irow, indices] /= a[irow, i]
            for j in range(irow) + range(irow+1, row):
                a[j, indices] -= a[irow, indices] / a[irow, i] * a[j,i]
            irow += 1
        else:
            independent.append(i)
    param = np.zeros((column, len(independent)), dtype=np.float)
    if len(independent) >0:
        for i, de in enumerate(dependent):
            for j, ind in enumerate(independent):
                param[de, j] = -a[i, ind]
        for i, ind in enumerate(independent):
            param[ind, i] = 1
    return a, param, independent