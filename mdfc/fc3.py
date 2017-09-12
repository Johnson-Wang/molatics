__author__ = 'xinjiang'
import numpy as np
from mdfc.fcmath import gaussian_py, similarity_transformation, mat_dot_product, gaussian
from itertools import permutations
from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors
from fc2 import get_index_of_atom

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

def get_fc3_coefficients(triplets, symmetry): # returning the fc3 coefficients at a specific triplet
    positions = symmetry.positions
    natom = len(positions)
    coefficients = np.zeros((natom, natom, natom), dtype='intc')
    ifc_map = np.zeros_like(coefficients)
    s2u_map = symmetry.cell.get_supercell_to_unitcell_map()
    for triplet in np.ndindex(natom, natom, natom):
        if triplet[0] not in np.unique(s2u_map):
            continue
        for i, (atom1, atom2, atom3) in enumerate(permutations(triplet)):
            atom1_ = symmetry.mapping[atom1]
            nope = symmetry.mapping_operations[atom1]
            rot1 = symmetry.rotations[nope]
            atom2_ = symmetry.get_atom_sent_by_operation(atom2, nope)
            atom2_, rot2 = symmetry.get_atom_mapping_under_sitesymmetry(atom1_, atom2_)
            atom3_ = symmetry.get_atom_sent_by_operation(atom3, nope)
            atom3_ = symmetry.get_atom_sent_by_site_sym(atom1_, atom3_, rot2)
            bond_symmetry2 = symmetry.get_site_symmetry_at_atoms([atom1_, atom2_])
            atom3_, rot3 = symmetry.get_atom_mapping_under_sitesymmetry(atom1_, atom3_, bond_symmetry2)
            if (atom1_, atom2_, atom3_) in triplets:
                rot = symmetry.rot_inverse(symmetry.rot_multiply(rot3, symmetry.rot_multiply(rot2, rot1)))
                coefficients[tuple(triplet)] =\
                    symmetry.tensor3[rot + i * len(symmetry.pointgroup_operations)]
                ifc_map[tuple(triplet)] = triplets.index(triplet)

    for atom1 in np.arange(natom):
        if atom1 in np.unique(s2u_map):
            continue
        atom1_ = s2u_map[atom1]
        disp = positions[atom1] - positions[atom1_]
        for atom2, atom3 in np.ndindex((natom, natom)):
            pos_atom2 = positions[atom2] - disp
            atom2_ = get_index_of_atom(pos_atom2, positions, symmetry.symprec)
            pos_atom3 = positions[atom3] - disp
            atom3_ = get_index_of_atom(pos_atom3, positions, symmetry.symprec)
            ifc_map[atom1, atom2, atom3] = ifc_map[atom1_, atom2_, atom3_]
            coefficients[atom1, atom2, atom3] = coefficients[atom1_, atom2_, atom3_]
    return coefficients, ifc_map


def get_fc3_spg_invariance(triplets,
                           symmetry):
    "Find all spg symmetries that map the triplet to itself and thus the symmetry would act as constraints"

    independents = []
    transforms = []
    for itriplet, triplet in enumerate(triplets):
        CC = [np.zeros(27)]
        bond_symmetry = symmetry.get_site_symmetry_at_atoms(triplet)
        invariant_transforms = []
        for i, (atom1, atom2, atom3) in enumerate(permutations(triplet)): # Leave out the original triplet
            atom1_ = symmetry.mapping[atom1]
            if not atom1_ == triplet[0]:
                continue
            nope = symmetry.mapping_operations[atom1]
            rot1 = symmetry.rotations[nope]
            atom2_ = symmetry.get_atom_sent_by_operation(atom2, nope)
            atom2_, rot2 = symmetry.get_atom_mapping_under_sitesymmetry(atom1_, atom2_)
            if not atom2_ == triplet[1]:
                continue
            atom3_ = symmetry.get_atom_sent_by_operation(atom3, nope)
            atom3_ = symmetry.get_atom_sent_by_site_sym(atom1_, atom3_, rot2)
            bond_symmetry2 = symmetry.get_site_symmetry_at_atoms([atom1_, atom2_])
            atom3_, rot3 = symmetry.get_atom_mapping_under_sitesymmetry(atom1_, atom3_, bond_symmetry2)
            if not atom3_ == triplet[2]:
                continue
            rot = symmetry.rot_inverse(symmetry.rot_multiply(rot3, symmetry.rot_multiply(rot2, rot1)))
            invariant_transforms.append(symmetry.tensor3[rot+i*len(symmetry.pointgroup_operations)])
        invariant_transforms = np.concatenate(invariant_transforms, axis=0)
        invariant_transforms = invariant_transforms - np.eye(27)
        for trans in invariant_transforms:
            is_found = False
            if not (np.abs(trans) < symmetry.symprec).all():
                row = trans / np.abs(trans).max()
                for j in np.arange(len(CC)):
                    if (np.abs(row - CC[j]) < symmetry.symprec).all():
                        is_found = True
                        break
                if not is_found:
                    CC.append(row)
        DD = np.array(CC, dtype='double')
        CC, transform, independent = gaussian(DD)
        independents.append(independent)
        transforms.append(transform)
    return independents, np.concatenate(transforms, axis=0)

    # if is_disperse:
    #     from scipy.sparse import coo_matrix
    #     trans = []
    #     for i, trip in enumerate(triplets_dict):
    #         start = sum(num_irred[:i])
    #         length = num_irred[i]
    #         transform_tmp = np.zeros((27, sum(num_irred)), dtype="float")
    #         transform_tmp[:, start:start + length] = trip['transform']
    #         non_zero = np.where(np.abs(transform_tmp) > symprec)
    #         transform_tmp_sparse = coo_matrix((transform_tmp[non_zero], non_zero), shape=transform_tmp.shape)
    #         trans.append(transform_tmp_sparse)
    # else:
    #     trans = np.zeros((len(triplets_dict), 27, sum(num_irred)), dtype="float")
    #     for i, trip in enumerate(triplets_dict):
    #         start = sum(num_irred[:i])
    #
    #         length = num_irred[i]
    #         trans[i,:, start:start + length] = trip['transform']


def get_fc3_translational_invariance(supercell,
                                     trans,
                                     coeff,
                                     ifc_map,
                                     precision=1e-6):
    # set translational invariance
    unit_atoms = np.unique(supercell.get_supercell_to_unitcell_map())
    natom = supercell.get_number_of_atoms()
    num_irred = trans[0].shape[-1]
    ti_transforms =[np.zeros(num_irred, dtype='double')]
    for i, atom1 in enumerate(unit_atoms):
        for atom2 in range(natom):
            ti_transform = np.zeros((27, num_irred), dtype='double')
            for atom3 in range(natom):
                irred_triplet = ifc_map[i, atom2, atom3]
                transform = trans[irred_triplet]
                ti_transform += mat_dot_product(coeff[i, atom2, atom3], transform, is_sparse=True) # transform maps from irreducible elements while coeff maps from irreducible triplets
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
        CC, transform, independent = gaussian_py(np.array(ti_transforms, dtype='double'), prec=precision)
    return independent, transform


def get_trim_fc3(supercell,
                 trans,
                 triplets_reduced,
                 symprec,
                 precision=1e-5,
                 triplets_included=None,
                 is_trim_boundary=False):
    num_irred = trans[0].shape[-1]
    unit_atoms = np.unique(supercell.get_supercell_to_unitcell_map())
    zero_fc3s =[np.zeros(num_irred, dtype='double')]
    for i, (atom1, atom2, atom3) in enumerate(triplets_reduced):
        first = np.where(atom1 == unit_atoms)[0][0]
        is_trim = False
        if triplets_included is not None and not triplets_included[i]:
            is_trim = True
        if is_trim_boundary:
            dist12 = get_equivalent_smallest_vectors(atom2, atom1, supercell, supercell.get_cell(), symprec=symprec)
            dist23 = get_equivalent_smallest_vectors(atom3, atom2, supercell, supercell.get_cell(), symprec=symprec)
            dist13 = get_equivalent_smallest_vectors(atom3, atom1, supercell, supercell.get_cell(), symprec=symprec)
            if len(dist12) > 1 or len(dist23) > 1 or len(dist13) > 1:
                is_trim = True
        if is_trim:
            import scipy.sparse
            if scipy.sparse.issparse(trans[i]):
                zero_fc3 = trans[i].toarray()
            else:
                zero_fc3 = trans[i]
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
        CC, transform, independent = gaussian_py(np.array(zero_fc3s, dtype='double'), prec=precision)
    return independent, transform

def get_fc3_rotational_invariance(fc2, supercell, trans, coeff, ifc_map, symprec=1e-5, precision=1e-6):
    "Returning the constraint equations to be used in Lagrangian Multipliers"
    precision *= 1e2
    unit_atoms = np.unique(supercell.get_supercell_to_unitcell_map())
    natom = supercell.get_number_of_atoms()
    num_irred = trans[0].shape[-1]
    lattice = supercell.get_cell()
    eijk = np.zeros((3,3,3), dtype="intc")
    eijk[0,1,2] = eijk[1,2,0] = eijk[2,0,1] = 1
    eijk[0,2,1] = eijk[2,1,0] = eijk[1,0,2] = -1 # epsilon matrix, which is an antisymmetric 3 * 3 * 3 tensor
    torques =[np.zeros(num_irred + 1)] # considering the constant column
    for i, atom1 in enumerate(unit_atoms):
        for atom2 in range(natom):
            torque = np.zeros((27, num_irred), dtype=np.float)
            t2_1 = np.einsum("cb, cav -> abv", fc2[atom1, atom2], eijk).flatten()
            t2_2 = np.einsum("ac, cbv -> abv", fc2[atom1, atom2], eijk).flatten()
            torque_fc2 = t2_1 + t2_2
            for atom3 in range(natom):
                fc_temp = mat_dot_product(coeff[i, atom2, atom3], trans[ifc_map[i, atom2, atom3]], is_sparse=True).reshape(3, 3, 3, -1)
                vectors = get_equivalent_smallest_vectors(atom3,
                                                          atom1,
                                                          supercell,
                                                          lattice,
                                                          symprec)
                r_frac = np.array(vectors).sum(axis=0) / len(vectors)
                r = np.dot(r_frac, lattice)
                disp = np.einsum("j, ijk -> ik", r, eijk)
                t3 = np.einsum("abcN, cv -> abvN", fc_temp, disp).reshape(27, num_irred)
                torque += t3
            torque = np.hstack((torque, -torque_fc2[:, np.newaxis])) #negative sign means Bx = d
            for k in range(27):
                if not (np.abs(torque[k])< precision).all():
                    argmax = np.argmax(np.abs(torque[k]))
                    torque[k] /= torque[k, argmax]
                    is_exist = np.all(np.abs(torque[k] - np.array(torques)) < precision, axis=1)
                    if (is_exist == False).all():
                        torques.append(torque[k] / torque[k, argmax])
    print "Number of constraints of IFC3 from rotational invariance:%d"%(len(torques) - 1)
    torques = np.array(torques)[1:]
    return torques[:, :-1], torques[:, -1] # return B and d

