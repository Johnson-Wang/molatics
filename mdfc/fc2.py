__author__ = 'xinjiang'
from itertools import permutations
import numpy as np
from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors
from mdfc.fcmath import  gaussian, mat_dot_product


def get_index_of_atom(pos_atom, positions, symprec=1e-6):
    diff = pos_atom - positions
    diff -= np.rint(diff)
    where,  = np.where(np.all(np.abs(diff) < symprec, axis=-1))
    if len(where) == 0:
        return
    return where[0]

def get_indices_of_atoms(pos_atoms, positions, symprec=1e-6):
    diff = pos_atoms[:, np.newaxis, :] - positions[np.newaxis, :, :] # shape[natom_pos, natom, 3]
    diff -= np.rint(diff)
    _, where = np.where(np.all(np.abs(diff) < symprec, axis=-1)) # shape: natom_pos
    assert len(where) == len(pos_atoms)
    return where

def get_fc2_coefficients(pairs, symmetry):
    natom = len(symmetry.positions)
    positions = symmetry.positions
    s2u_map = symmetry.cell.get_supercell_to_unitcell_map()
    coeff = np.zeros((natom, natom), dtype="intc")
    ifc_map = np.zeros((natom, natom), dtype="intc")
    #Consider atom1 in the first unit-cell
    first_cell = np.unique(s2u_map)
    for atom1, atom2 in np.ndindex((natom, natom)):
        if atom1 not in first_cell:
            continue
        ifc_map[atom1, atom2], coeff[atom1, atom2] = get_fc2_coefficient_single(pairs, symmetry, atom1, atom2)
    #Other pairs can be directly mapped using translational symmetry
    for atom1 in np.arange(natom):
        if atom1 in first_cell:
            continue
        atom1_ = s2u_map[atom1]
        disp = positions[atom1] - positions[atom1_]
        pos_atom2 = positions - disp # shape:[natom, 3]
        atom2_ = get_indices_of_atoms(pos_atom2, positions, symmetry.symprec)
        ifc_map[atom1, :] = ifc_map[atom1_, atom2_]
        coeff[atom1, :] = coeff[atom1_, atom2_]
    return ifc_map, coeff

def get_fc2_coefficient_single(pairs, symmetry, atom1=0, atom2=0):
    for i, (_atom1, _atom2) in enumerate(permutations([atom1, atom2])):
        nope = symmetry.mapping_operations[atom1]
        rot1 = symmetry.rotations[nope]
        atom1_ = symmetry.mapping[atom1] # first irreducible atom
        atom2_ = symmetry.get_atom_sent_by_operation(_atom2, nope) # second atom should move with the first one
        atom2_, rot2 = symmetry.get_atom_mapping_under_sitesymmetry(atom1_, atom2_) # second irreducible atom
        if (atom1_, atom2_) not in pairs:
            continue # move to the transposed pair
        ifc_map = pairs.index((atom1_, atom2_))
        rot = symmetry.rot_multiply(rot2, rot1) # rot=rot2.rot1
        coeff = rot + i * len(symmetry.pointgroup_operations)
        return ifc_map, coeff

def get_fc2_spg_invariance(pairs, symmetry):
    """Obtain the invariance under space group symmetries, i.e.
    if R.Phi.R^T = Phi (or R.Phi^T.R^T for permutational case),
    where R is a rotational matrix, then components of Phi
    can be reduced using R by Gaussian elimination method
    """
    trans = []
    indeps = []
    npgope = len(symmetry.pointgroup_operations)
    tensor2 = symmetry.tensor2
    for i,pair in enumerate(pairs):
        atom1, atom2 = pair
        bond_symmetry = symmetry.get_site_symmetry_at_atoms([atom1, atom2])

        invariant_transforms = tensor2[bond_symmetry]
        # find all symmetries that keep the bond atom1-atom2 invariant
        ###############For permutations#######################
        if symmetry.mapping[atom2] == atom1:
            # this requires the bond atom2-atom1 can be mapped to atom1-atom2
            nope = symmetry.mapping_operations[atom2] #the index of operations that sent atom2 to atom1
            rot1 = symmetry.rotations[nope]
            atom2_ = symmetry.get_atom_sent_by_operation(atom1, nope)
            atom2_, rot2 = symmetry.get_atom_mapping_under_sitesymmetry(atom1, atom2_)
            if atom2_ == atom2:
                rot = symmetry.rot_multiply(rot2, rot1)
                bond_symmetry2 = [symmetry.rot_multiply(sym, rot)+npgope
                                  for sym in bond_symmetry]
                #R.P.12 = 12, where P is a permutation matrix and R is a rotational matrix
                invariant_transforms =\
                    np.concatenate((invariant_transforms, tensor2[bond_symmetry2]), axis=0)
        invariant_transforms -= np.eye(9)
        invariant_transforms = invariant_transforms.reshape(-1,9)
        CC, transform, independent = gaussian(invariant_transforms)
        trans.append(transform)
        indeps.append(independent)
    return indeps, np.hstack(trans)

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
