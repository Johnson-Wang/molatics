__author__ = 'xinjiang'
import numpy as np
from itertools import permutations
from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors
from mdfc.fcmath import similarity_transformation, gaussian, mat_dot_product
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
    transform[:] = np.eye(9) # An identity matrix is broadcast
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
            trans_cart = similarity_transformation(lattice, transf)
            trans_tensor = np.kron(trans_cart.T, trans_cart.T).reshape(3,3,9) # from irreducible to general mapping
            trans_permute = trans_tensor.swapaxes(0,1).reshape(9,9) # permutation of the original fc2
            # transf = np.einsum("ij,km -> kijm", trans_cart, trans_cart).reshape(9,9)
            transform[i] = np.dot(trans_permute, transform[index_pair])
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
            R = np.dot(r2, r1) # from original to irreducible pair
            R_cart = np.double(similarity_transformation(lattice, R))
            # coeff_temp = np.einsum("ij, kl -> ikjl", R_cart, R_cart).reshape(9,9)
            coeff_temp = np.kron(R_cart.T, R_cart.T)
            index_orig = pairs_orig.index((a1, a2))
            star2 = pairs_orig[mapping_pair[index_orig]]
            ifc_map[atom1, atom2] = pairs_reduced.index(star2)
            permu_trans = transform_pair[index_orig]
            coeff_temp = np.dot(coeff_temp, permu_trans)
            #take the permutation matrix into consideration
            coeff[atom1, atom2] = coeff_temp[:] # inverse equals transpose
    return coeff, ifc_map

def get_fc2_spg_invariance(pairs_orig,
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
                        if (np.abs(r-rot) < symprec).all():
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
                        if not (np.abs(BB[i]) < symprec).all():
                            for j in np.arange(len(CC)):
                                if (np.abs(BB[i] - CC[j]) < symprec).all():
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
    return np.hstack((dic['independent'] for dic in doublets_dict)), trans

def get_fc2_translational_invariance(supercell, trans, coeff, ifc_map, precision=1e-6):
    natom = supercell.get_number_of_atoms()
    unit_atoms = np.unique(supercell.get_supercell_to_unitcell_map())
    num_irred = trans.shape[-1]
    ti_transforms =[np.zeros(num_irred)]
    for i, atom1 in enumerate(unit_atoms):
        ti_transform = np.zeros((9, num_irred))
        for atom2 in np.arange(natom):
            irred_doublet = ifc_map[atom1, atom2]
            ti_transform += np.dot(coeff[atom1, atom2], trans[irred_doublet])
        for k in range(9):
            if not (np.abs(ti_transform[k])< precision).all():
                # ti_transform_row = ti_transform[k] / np.abs(ti_transform[k]).max()
                # ti_transforms.append(ti_transform_row)
                argmax = np.argmax(np.abs(ti_transform[k]))
                ti_transform[k] /= ti_transform[k, argmax]
                is_exist = np.all(np.abs(ti_transform[k] - np.array(ti_transforms)) < precision, axis=1)
                if (is_exist == False).all():
                    ti_transforms.append(ti_transform[k] / ti_transform[k, argmax])
    print "Number of constraints of fc2 from translational invariance:%d"% (len(ti_transforms) - 1)
    CC, transform, independent = gaussian(np.array(ti_transforms), precision)
    return independent, transform

def get_trim_fc2(supercell,
                 trans,
                 coeff,
                 ifc_map,
                 symprec,
                 precision=1e-5,
                 pairs_included=None,
                 is_trim_boundary=False):
    unit_atoms = np.unique(supercell.get_supercell_to_unitcell_map())
    natom = supercell.get_number_of_atoms()
    num_irred = trans.shape[-1]
    ti_transforms =[np.zeros(num_irred)]
    for i, atom1 in enumerate(unit_atoms):
        for atom2 in np.arange(natom):
            is_trim = False
            if pairs_included is not None and not pairs_included[ifc_map[atom1, atom2]]:
                is_trim = True
            if is_trim_boundary:
                dist = get_equivalent_smallest_vectors(atom2, atom1, supercell, supercell.get_cell(), symprec=symprec)
                if len(dist) > 1:
                    is_trim = True
            if is_trim:
                irred_doublet = ifc_map[atom1, atom2]
                ti_transform = np.dot(coeff[atom1, atom2], trans[irred_doublet])
                for k in range(9):
                    if not (np.abs(ti_transform[k])< precision).all():
                        argmax = np.argmax(np.abs(ti_transform[k]))
                        ti_transform[k] /= ti_transform[k, argmax]
                        is_exist = np.all(np.abs(ti_transform[k] - np.array(ti_transforms)) < precision, axis=1)
                        if (is_exist == False).all():
                            ti_transforms.append(ti_transform[k] / ti_transform[k, argmax])

    print "Number of constraints of fc2 from a cutoff of interaction distance:%d"%len(ti_transforms)
    CC, transform, independent = gaussian(np.array(ti_transforms), precision)
    return independent, transform

def get_fc2_rotational_invariance(supercell,
                                  trans,
                                  coeff,
                                  ifc_map,
                                  symprec,
                                  precision=1e-8,
                                  is_Huang=True): #Gazis-Wallis invariance
    positions = supercell.get_scaled_positions()
    unit_atoms = np.unique(supercell.get_supercell_to_unitcell_map())
    natom = supercell.get_number_of_atoms()
    num_irred2 = trans.shape[-1]
    lattice = supercell.get_cell()
    eijk = np.zeros((3,3,3), dtype="intc")
    eijk[0,1,2] = eijk[1,2,0] = eijk[2,0,1] = 1
    eijk[0,2,1] = eijk[2,1,0] = eijk[1,0,2] = -1 # epsion matrix, which is an antisymmetric 3 * 3 * 3 tensor
    torques =[np.zeros(num_irred2)]
    for i, patom_num in enumerate(unit_atoms):
        force = np.zeros((27, num_irred2), dtype='double')
        for j in range(natom):
            fc_temp = np.dot(coeff[patom_num, j], trans[ifc_map[patom_num, j]]).reshape(3,3,-1)
            vectors = get_equivalent_smallest_vectors(j,
                                                      patom_num,
                                                      supercell,
                                                      lattice,
                                                      symprec)
            r_frac = np.array(vectors).sum(axis=0) / len(vectors)
            r = np.dot(r_frac, lattice)
            force_tmp = np.einsum('abN, c->abcN', fc_temp, r) - np.einsum('acN, b->abcN', fc_temp, r)
            force += force_tmp.reshape(27, -1)
        for k in range(27):
            if not (np.abs(force[k])< precision).all():
                argmax = np.argmax(np.abs(force[k]))
                force[k] /= force[k, argmax]
                is_exist = np.all(np.abs(force[k] - np.array(torques)) < precision, axis=1)
                if (is_exist == False).all():
                    torques.append(force[k] / force[k, argmax])
    print "Number of constraints of fc2 from rotational invariance:%d"%(len(torques)-1)
    CC, transform, independent = gaussian(np.array(torques), precision)
    if is_Huang:
        precision *= 1e2
        print "The Born-Huang invariance condition is also included in rotational symmetry"
        trans2 = mat_dot_product(trans, transform, is_sparse=True)
        num_irred2 = trans2.shape[-1]
        torques =[np.zeros(num_irred2)]
        unit_atoms = np.unique(supercell.get_supercell_to_unitcell_map())
        torque_tmp = np.zeros((9, 9, num_irred2), dtype='double')
        for patom_num in unit_atoms:
            for j in range(natom):
                fc_temp = np.dot(coeff[patom_num, j], trans2[ifc_map[patom_num, j]])
                vectors12 = get_equivalent_smallest_vectors(j,
                                          patom_num,
                                          supercell,
                                          lattice,
                                          symprec)
                for r12 in vectors12:
                    v12 = np.dot(r12, lattice)
                    v12_outer = np.kron(v12, v12)
                    torque_tmp += np.einsum('iN, j->ijN', fc_temp, v12_outer) / len(vectors12)
        torque_tmp = torque_tmp - np.swapaxes(torque_tmp, 0, 1)
        torque_tmp = torque_tmp.reshape(-1, num_irred2)
        for i in range(81):
            if not (np.abs(torque_tmp[i])< precision).all():
                argmax = np.argmax(np.abs(torque_tmp[i]))
                torque_tmp[i] /= torque_tmp[i, argmax]
                is_exist = np.all(np.abs(torque_tmp[i] - np.array(torques)) < precision, axis=1)
                if (is_exist == False).all():
                    torques.append(torque_tmp[i] / torque_tmp[i, argmax])
        print "Number of constraints of fc2 from Born-Huang invariance condition:%d"%(len(torques)-1)
        CC, transform_gw, independent_gw = gaussian(np.array(torques), precision)

        independent = np.array([independent[i] for i in independent_gw])
        transform = np.dot(transform,transform_gw)
    return independent, transform

def get_fc2_coefficient_and_mapping(doublets_reduced, pairs, symmetry, cell):
    natom = cell.get_number_of_atoms()
    positions = cell.get_scaled_positions()
    coeff = np.zeros((natom, natom, 9, 9), dtype=np.float)
    ifc_map = np.zeros((natom, natom), dtype="intc")
    lattice = cell.get_cell().T
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
            R_cart = np.double(similarity_transformation(lattice, R))
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

def show_drift_force_constants(force_constants, name="force constants"):
    num_atom = force_constants.shape[0]
    maxval1 = 0
    maxval2 = 0
    for i, j, k in list(np.ndindex((num_atom, 3, 3))):
        val1 = force_constants[:, i, j, k].sum()
        val2 = force_constants[i, :, j, k].sum()
        if abs(val1) > abs(maxval1):
            maxval1 = val1
        if abs(val2) > abs(maxval2):
            maxval2 = val2
    print ("max drift of %s:" % name), maxval1, maxval2

def show_rotational_invariance(force_constants,
                               supercell,
                               symprec=1e-5,
                               log_level=1):
    """
    *** Under development ***
    Just show how force constant is close to the condition of rotational invariance,
    """
    print "Check rotational invariance ..."

    fc = force_constants
    unit_atoms = np.unique(supercell.get_supercell_to_unitcell_map())
    volume_unitcell = supercell.get_volume() * len(unit_atoms) / supercell.get_number_of_atoms()
    abc = "xyz"
    eijk = np.zeros((3,3,3), dtype="intc")
    eijk[0,1,2] = eijk[1,2,0] = eijk[2,0,1] = 1
    eijk[0,2,1] = eijk[2,1,0] = eijk[1,0,2] = -1 # epsilon matrix, which is an antisymmetric 3 * 3 * 3 tensor

    stress = np.zeros((3, 3), dtype='double')
    for pi, p in enumerate(unit_atoms):
        for i in range(3):
            mat = np.zeros((3, 3), dtype='double')
            for s in range(supercell.get_number_of_atoms()):
                vecs = np.array(get_equivalent_smallest_vectors(
                        s, p, supercell, supercell.get_cell(), symprec))
                m = len(vecs)
                v = np.dot(vecs[:,:].sum(axis=0) / m, supercell.get_cell())
                for j in range(3):
                    for k in range(3):
                        mat[j, k] += (fc[p, s, i, j] * v[k] -
                                      fc[p, s, i, k] * v[j])
            stress += np.abs(mat)
            if log_level == 2:
                print "Atom %d %s" % (p+1, abc[i])
                for vec in mat:
                    print "%10.5f %10.5f %10.5f" % tuple(vec)
    if log_level == 1:
        print "System stress residue enduced by rigid-body rotations(eV/A)"
        for vec in stress:
            print "%10.5f %10.5f %10.5f" % tuple(vec)

    ElasticConstants = np.zeros((3,3,3,3), dtype='double')
    for s1 in unit_atoms:
        for s2 in range(supercell.get_number_of_atoms()):
            vec12s = np.array(get_equivalent_smallest_vectors(
                    s2, s1, supercell, supercell.get_cell(), symprec))
            vec12s = np.dot(vec12s, supercell.get_cell())
            for v12 in vec12s:
                ElasticConstants += -np.einsum('ij, k, l->ijkl', fc[s1, s2], v12, v12)  / len(vec12s) / 2.
    ElasticConstants = ElasticConstants.reshape(9,9)
    non_sym_tensor = ElasticConstants - ElasticConstants.T
    if log_level == 2:
        print 'Born-Huang rotational invariance condition (eV)'
        for i in range(9):
            print "%10.5f " * 9 %tuple(non_sym_tensor[i])
    elif log_level == 1:
        M_sum = np.abs(non_sym_tensor).sum()
        print 'Born-Huang rotational invariance condition (eV): %10.5f' %M_sum
