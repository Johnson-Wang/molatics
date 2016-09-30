__author__ = 'xinjiang'
import numpy as np
from mdfc.fc2 import get_pairs, get_next_atom, get_atom_sent_by_operation, get_atom_sent_by_site_sym,\
    get_all_operations_at_star, get_rotations_at_star, get_operations_at_star, similarity_transformation
from mdfc.fcmath import gaussian, similarity_transformation
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
    transform = np.zeros((len(triplets), 27, 27), dtype='double')
    transform[:] += np.eye(27)
    indatoms1 = np.unique(mappings1)
    for itriplet, (p1, p2, p3) in enumerate(triplets):
        triplet = (p1, p2, p3)
        # if triplet == (0, 5, 15):
        #     print "good"
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
                transf_cart = similarity_transformation(lattice, transf).T # Coordinate transform from  irreducible to general
                seq = "".join(['ikm'[permute.index(i)] for i in range(3)]) # find the original index before permutation
                transform_temp = np.einsum("ij, kl, mn -> %sjln" %seq, transf_cart, transf_cart, transf_cart).reshape(27, 27)
                transform[itriplet] = np.dot(transform_temp, transform[permuted_index]) # from irreducible to general
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
                iratom3 = get_atom_sent_by_operation(atom3, positions, rot1, tran1, symprec=symprec)
                iratom3 = get_atom_sent_by_site_sym(iratom3, iratom1, positions, rot2, symprec=symprec)
                rot3 = rotations3[index1, index2, map_ope3[index1, index2, iratom3]]
                iratom3 =mappings3[index1, index2, iratom3]
                rot = np.dot(rot3, np.dot(rot2, rot1))
                #rot transforms an arbitrary triplet to the irreducible one
                index_triplet = triplets_orig.index((iratom1, iratom2, iratom3))
                star3 = triplets_orig[mapping_triplet[index_triplet]]
                ifc_map[atom1, atom2, atom3] = triplets_reduced.index(star3)
                rot_cart = np.double(similarity_transformation(lattice, rot)).T # from irreducible to general
                coeff_temp = np.kron(np.kron(rot_cart, rot_cart), rot_cart)
                coeff[atom1, atom2, atom3] = np.dot(coeff_temp, transform_triplet[index_triplet])
                # Considering the permutation symmetry previously obtained
    return  coeff, ifc_map

def get_fc3_spg_invariance(triplets,
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
    "Find all spg symmetries that map the triplet to itself and thus the symmetry would act as constraints"
    triplets_dict = []
    iatoms1 = np.unique(mappings1)
    for itriplet, triplet in enumerate(triplets):
        triplet_dict = {}
        triplet_dict["triplet"] = triplet
        CC = [np.zeros(27)]

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
                #get all the symmetry operations that keep atom1 unchanged
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
                    #get all the symmetry operations that keep both atom1 and atom2 unchanged
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
                            if (np.abs(r-rot) < symprec).all():
                                isfound = True
                                break
                        if not isfound:
                            rot_all.append(rot)
                        else:
                            continue
                        # rot_cart = np.double(similarity_transformation(lattice, rot))
                        # seq = "".join(["ikm"[i] for i in permute])
                        # PP  = np.einsum("ij,kl,mn -> %sjln"%seq, rot_cart, rot_cart, rot_cart).reshape(27, 27).T

                        rot_cart = np.double(similarity_transformation(lattice, rot)).T
                        seq = "".join(['ikm'[permute.index(i)] for i in range(3)]) # find the original index before permutation
                        PP = np.einsum("ij,kl,mn -> %sjln"%seq, rot_cart, rot_cart, rot_cart).reshape(27, 27)
                        BB = PP - np.eye(27)
                        for i in np.arange(27):
                            is_found = False
                            if not (np.abs(BB[i]) < symprec).all():
                                row = BB[i] / np.abs(BB[i]).max()
                                for j in np.arange(len(CC)):
                                    if (np.abs(row - CC[j]) < symprec).all():
                                        is_found = True
                                        break
                                if not is_found:
                                    CC.append(row)
        DD = np.array(CC, dtype='double')
        CC, transform, independent = gaussian(DD)
        # tt = np.zeros((27, 27), dtype='double')
        # ii = np.zeros(27, dtype='intc')
        # import _mdfc
        # nn = _mdfc.gaussian(tt, DD, ii, symprec)
        # transform = tt[:, :nn]
        # independent = ii[:nn]

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
    ind_elements =  np.hstack((dic['independent'] for dic in triplets_dict))
    return ind_elements, trans

def get_fc3_translational_invariance(supercell,
                                     trans,
                                     coeff,
                                     ifc_map,
                                     precision=1e-6):
    # set translational invariance
    unit_atoms = np.unique(supercell.get_supercell_to_unitcell_map())
    natom = supercell.get_number_of_atoms()
    num_irred = trans.shape[-1]
    ti_transforms =[np.zeros(num_irred, dtype='double')]
    for i, atom1 in enumerate(unit_atoms):
        for atom2 in range(natom):
            ti_transform = np.zeros((27, num_irred), dtype='double')
            for atom3 in range(natom):
                irred_triplet = ifc_map[atom1, atom2, atom3]
                transform = trans[irred_triplet]
                ti_transform += np.dot(coeff[atom1, atom2, atom3], transform) # transform maps from irreducible elements while coeff maps from irreducible triplets
            for k in range(27):
                if not (np.abs(ti_transform[k])< precision).all():
                    argmax = np.argmax(np.abs(ti_transform[k]))
                    ti_transform[k] /= ti_transform[k, argmax]
                    is_exist = np.all(np.abs(ti_transform[k] - np.array(ti_transforms)) < precision, axis=1)
                    if (is_exist == False).all():
                        ti_transforms.append(ti_transform[k] / ti_transform[k, argmax])

    print "Number of constraints of fc3 from translational invariance:%d"%(len(ti_transforms) - 1)
    try:
        import _mdfc
        transform = np.zeros((num_irred, num_irred), dtype='double')
        independent = np.zeros(num_irred, dtype='intc')
        num_independent = _mdfc.gaussian(transform, np.array(ti_transforms, dtype='double'), independent, precision)
        transform = transform[:, :num_independent]
        independent = independent[:num_independent]
    except ImportError:
        CC, transform, independent = gaussian(np.array(ti_transforms, dtype='double'), prec=precision)
    return independent, transform


def get_trim_fc3(supercell,
                 trans,
                 coeff,
                 ifc_map,
                 symprec,
                 precision=1e-5,
                 triplets_included=None,
                 is_trim_boundary=False):
    unit_atoms = np.unique(supercell.get_supercell_to_unitcell_map())
    natom = supercell.get_number_of_atoms()
    num_irred = trans.shape[-1]
    zero_fc3s =[np.zeros(num_irred, dtype='double')]
    for i, atom1 in enumerate(unit_atoms):
        for atom2 in np.arange(natom):
            for atom3 in np.arange(natom):
                is_trim = False
                if triplets_included is not None and not triplets_included[ifc_map[atom1, atom2, atom3]]:
                    is_trim = True
                if is_trim_boundary:
                    dist12 = get_equivalent_smallest_vectors(atom2, atom1, supercell, supercell.get_cell(), symprec=symprec)
                    dist23 = get_equivalent_smallest_vectors(atom3, atom2, supercell, supercell.get_cell(), symprec=symprec)
                    dist13 = get_equivalent_smallest_vectors(atom3, atom1, supercell, supercell.get_cell(), symprec=symprec)
                    if len(dist12) > 1 or len(dist23) > 1 or len(dist13) > 1:
                        is_trim = True
                if is_trim:
                    irred_doublet = ifc_map[atom1, atom2, atom3]
                    zero_fc3 = np.dot(coeff[atom1, atom2, atom3], trans[irred_doublet])
                    for k in range(27):
                        if not (np.abs(zero_fc3[k])< precision).all():
                            argmax = np.argmax(np.abs(zero_fc3[k]))
                            zero_fc3[k] /= zero_fc3[k, argmax]
                            is_exist = np.all(np.abs(zero_fc3[k] - np.array(zero_fc3s)) < precision, axis=1)
                            if (is_exist == False).all():
                                zero_fc3s.append(zero_fc3[k] / zero_fc3[k, argmax])
    print "Number of constraints of fc3 from a cutoff of interaction distance:%d"% (len(zero_fc3s) - 1)
    try:
        import _mdfc
        transform = np.zeros((num_irred, num_irred), dtype='double')
        independent = np.zeros(num_irred, dtype='intc')
        num_independent = _mdfc.gaussian(transform, np.array(zero_fc3s, dtype='double'), independent, precision)
        transform = transform[:, :num_independent]
        independent = independent[:num_independent]
    except ImportError:
        CC, transform, independent = gaussian(np.array(zero_fc3s, dtype='double'), prec=precision)
    return independent, transform


def get_fc3_rotational_invariance(fc2, supercell, trans, coeff, ifc_map, symprec=1e-5, precision=1e-6):
    precision *= 1e3
    unit_atoms = np.unique(supercell.get_supercell_to_unitcell_map())
    natom = supercell.get_number_of_atoms()
    num_irred = trans.shape[-1]
    lattice = supercell.get_cell()
    eijk = np.zeros((3,3,3), dtype="intc")
    eijk[0,1,2] = eijk[1,2,0] = eijk[2,0,1] = 1
    eijk[0,2,1] = eijk[2,1,0] = eijk[1,0,2] = -1 # epsilon matrix, which is an antisymmetric 3 * 3 * 3 tensor
    torques =[np.zeros(num_irred)]
    for i, atom1 in enumerate(unit_atoms):
        for atom2 in range(natom):
            torque = np.zeros((27, num_irred), dtype=np.float)
            for atom3 in range(natom):
                fc_temp = np.dot(coeff[atom1, atom2, atom3], trans[ifc_map[atom1, atom2, atom3]]).reshape(3,3,3, -1)
                vectors = get_equivalent_smallest_vectors(atom3,
                                                          atom1,
                                                          supercell,
                                                          lattice,
                                                          symprec)
                r_frac = np.array(vectors).sum(axis=0) / len(vectors)
                r = np.dot(r_frac, lattice)
                disp = np.einsum("j, ijk -> ik", r, eijk)
                t3 = np.einsum("abcN, cv -> abvN", fc_temp, disp).reshape(27, num_irred)
                t2_1 = np.einsum("cb, cav -> abv", fc2[atom1, atom2], eijk).flatten()
                t2_2 = np.einsum("ac, cbv -> abv", fc2[atom1, atom2], eijk).flatten()
                torque += (t3.T + t2_1 + t2_2).T
            for k in range(27):
                if not (np.abs(torque[k])< precision).all():
                    argmax = np.argmax(np.abs(torque[k]))
                    torque[k] /= torque[k, argmax]
                    is_exist = np.all(np.abs(torque[k] - np.array(torques)) < precision, axis=1)
                    if (is_exist == False).all():
                        torques.append(torque[k] / torque[k, argmax])
    print "Number of constraints of IFC3 from rotational invariance:%d"%(len(torques) - 1)

    try:
        import _mdfc
        transform = np.zeros((num_irred, num_irred), dtype='double')
        independent = np.zeros(num_irred, dtype='intc')
        num_independent = _mdfc.gaussian(transform, np.array(torques, dtype='double'), independent, precision)
        transform = transform[:, :num_independent]
        independent = independent[:num_independent]
    except ImportError:
        CC, transform, independent = gaussian(np.array(torques), prec=precision)
    return independent, transform

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