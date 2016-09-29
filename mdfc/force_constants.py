import numpy as np
import sys
# from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors
from phonopy.structure.cells import get_supercell, get_reduced_bases
from fc2 import get_atom_mapping, get_pairs_with_permute, get_fc2_coefficients, get_fc2_spg_invariance,\
    get_fc2_translational_invariance, get_fc2_rotational_invariance, expand_fc2_conditions
from fc3 import get_bond_symmetry, get_irreducible_triplets_with_permute, get_fc3_coefficients, get_fc3_spg_invariance,\
    get_fc3_translational_invariance, get_fc3_rotational_invariance
from file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5, write_fc2_hdf5, write_fc3_hdf5
from realmd.information import timeit
import matplotlib.pyplot as plt
DEBUG = False

search_directions = np.array(list(np.ndindex((3,3,3)))) - 1

# Helper methods
def get_equivalent_smallest_vectors(atom_number_supercell,
                                    atom_number_primitive,
                                    supercell,
                                    primitive_lattice,
                                    symprec,
                                    is_return_index=False):
    # reduced_bases = get_reduced_bases(supercell.get_cell(), symprec)
    reduced_bases = supercell.get_cell()
    positions = np.dot(supercell.get_positions(),
                       np.linalg.inv(reduced_bases))

    # Atomic positions are confined into the lattice made of reduced bases.
    for pos in positions:
        pos -= np.rint(pos)

    p_pos = positions[atom_number_primitive]
    s_pos = positions[atom_number_supercell]

    # relative_pos = np.array(list(np.ndindex((3,3,3)))) - 1
    differencessp = s_pos - p_pos + search_directions
    distancessp = np.sqrt(np.sum(np.dot(differencessp, reduced_bases) ** 2, axis=-1))
    equivalent_directions = np.where(distancessp - np.min(distancessp) < symprec)[0]
    diffsp = differencessp[equivalent_directions]
    relative_scale = np.dot(reduced_bases,np.linalg.inv(primitive_lattice))
    smallest_vectors = np.dot(diffsp, relative_scale)


    # for i in (-1, 0, 1):
    #     for j in (-1, 0, 1):
    #         for k in (-1, 0, 1):
    #             # The vector arrow is from the atom in primitive to
    #             # the atom in supercell cell plus a supercell lattice
    #             # point. This is related to determine the phase
    #             # convension when building dynamical matrix.
    #             diff = s_pos + np.array([i, j, k]) - p_pos
    #             differences.append(diff)
    #             vec = np.dot(diff, reduced_bases)
    #             distances.append(np.linalg.norm(vec))

    # minimum = min(distances)
    # smallest_vectors = []
    # for i in range(27):
    #     if abs(minimum - distances[i]) < symprec:
    #         relative_scale = np.dot(reduced_bases,
    #                                 np.linalg.inv(primitive_lattice))
    #         smallest_vectors.append(np.dot(differences[i], relative_scale))
    if not is_return_index:
        return smallest_vectors
    else:
        return smallest_vectors, equivalent_directions


class ForceConstants():
    def __init__(self, supercell, primitive, symmetry, cutoff=None, precision=1e-8):
        self._symmetry = symmetry
        self._supercell = supercell
        self._primitive = primitive
        self._ind1 = None
        self._ind2 = None
        self._ind3 = None
        self._num_atom = len(supercell.get_scaled_positions())
        self._positions = supercell.get_scaled_positions()
        self._symprec = 1e-6
        self._precision=precision
        self._lattice = supercell.get_cell().T
        self._cutoff = cutoff
        self._pairs_reduced = None
        self._pairs_included = None
        self._fc2 = None
        self._fc2_read = None
        self._fc3_read = None
        self._fc3_irred = None
        self._fc2_irred  = None

    def set_fc2_read(self, fc2_filename, rdim=None):
        fc2_read = read_fc2_from_hdf5(filename=fc2_filename)
        if rdim is not None:
            natom = self._supercell.get_number_of_atoms()
            self._fc2_read = np.zeros((natom, natom, 3, 3), dtype="double")
            supercell_orig = get_supercell(self._primitive, rdim)
            natom_orig = supercell_orig.get_number_of_atoms()
            pos_orig = supercell_orig.get_scaled_positions()
            pos = self._supercell.get_scaled_positions()

            unit_map_orig = np.unique(supercell_orig.get_supercell_to_unitcell_map())
            unit_map = np.unique(self._supercell.get_supercell_to_unitcell_map())
            scaled_positions = self._supercell.get_scaled_positions()
            for i, ip in zip(unit_map, unit_map_orig):
                for jp in range(natom_orig):
                    vectors = get_equivalent_smallest_vectors(jp, ip, supercell_orig, supercell_orig.get_cell(), self._symprec)
                    for vec in vectors:
                        rpos = np.dot(np.dot(vec, supercell_orig.get_cell()), np.linalg.inv(self._supercell.get_cell()))
                        diff = scaled_positions[i] + rpos - scaled_positions
                        j = np.where(np.all(np.abs(diff - np.rint(diff)) < self._symprec, axis=-1))[0][0]
                        self._fc2_read[i, j] = fc2_read[ip, jp]

            s2u_map = self._supercell.get_supercell_to_unitcell_map()
            scaled_positions = self._supercell.get_scaled_positions()
            for i in range(natom):
                if i in s2u_map:
                    continue
                ip = s2u_map[i]
                diff = scaled_positions[i] - scaled_positions[ip]
                for j in range(natom):
                    disp = scaled_positions[j] - diff - scaled_positions
                    jp = np.where(np.all(np.abs(disp - np.rint(disp)) < self._symprec, axis=-1))[0][0]
                    self._fc2_read[i, j] = self._fc2_read[ip, jp]
        else:
            self._fc2_read = fc2_read

    def get_fc2_read(self):
        return self._fc2_read

    def get_fc2(self):
        return self._fc2

    def set_translational_rotational_invariance(self):
        """
        Translational invariance and rotational invariance are imposed. The algorithm
        is adopted from J. Carrete, W. Li, et al., Materials Research Letters 0, 1 (2016).
        The force constants are transformed from Cartesian coordinate (3N x 3N) to internal coordinates ((3N-6) x (3N-6))
        using lst method and then transformed back.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import lsqr
        import scipy
        import matplotlib.pyplot as plt
        supercell = self._supercell
        pos = supercell.get_positions()
        # lattice = get_reduced_bases(supercell.get_cell(), self._symprec)

        rel_pos = supercell.get_scaled_positions()
        natom = supercell.get_number_of_atoms()
        Jacobian = np.zeros((27, natom, 3, natom * 3 - 6), dtype='double')
        unique_atoms = np.unique(supercell.s2u_map)
        i0 = unique_atoms[0]
        p0 = pos[i0]
        rest = range(1,natom)
        if len(unique_atoms) >= 2:
            i1 = unique_atoms[1]
            p1 = pos[i1]
            rest.remove(i1)
            if len(unique_atoms) >= 3:
                i2 = unique_atoms[2]
                p2 = pos[i2]
                rest.remove(i2)
        else:
            i1 = 1
            p1=pos[i1]
            rest.remove(i1)
        if len(rest) == natom - 2:
            for i in rest:
                p2 = pos[i]
                i2= i
                if not (np.abs(np.cross(p0-p1, p0-p2)) < 1e-7).all(): # colinear
                    rest.remove(i)
                    break

        assert len(get_equivalent_smallest_vectors(i1, i0, supercell, supercell.get_cell(), self._symprec)) == 1
        assert len(get_equivalent_smallest_vectors(i2, i0, supercell, supercell.get_cell(), self._symprec)) == 1
        assert len(get_equivalent_smallest_vectors(i2, i1, supercell, supercell.get_cell(), self._symprec)) == 1

        d10 = get_equivalent_smallest_vectors(i1, i0, supercell, supercell.get_cell(), self._symprec)[0]
        d10 = np.dot(d10, supercell.get_cell())
        d20 = get_equivalent_smallest_vectors(i2, i0, supercell, supercell.get_cell(), self._symprec)[0]
        d20 = np.dot(d20, supercell.get_cell())
        d21 = get_equivalent_smallest_vectors(i2, i1, supercell, supercell.get_cell(), self._symprec)[0]
        d21 = np.dot(d21, supercell.get_cell())

        dist = lambda x: np.sqrt(np.sum(x ** 2))
        Jacobian[0, i0, :, 0] = d10 / dist(d10)
        Jacobian[0, i1, :, 0] = -Jacobian[0, i0, :, 0]
        Jacobian[0, i0, :, 1] = d20 / dist(d20)
        Jacobian[0,i2, :, 1] = - Jacobian[0, i0, :, 1]
        Jacobian[0,i1, :, 2] = d21 / dist(d21)
        Jacobian[0,i2, :, 2] = - Jacobian[0, i1, :, 2]
        I = np.eye(3)
        for index, j in enumerate(rest):
            i = index + 1

            for direct in range(27):
                rj0 = rel_pos[j] - rel_pos[i0] + search_directions[direct]
                # rj0 = get_equivalent_smallest_vectors(j, i0, supercell, supercell.get_cell(), self._symprec)
                dj0 = np.dot(rj0, supercell.get_cell())
                rj1 = rel_pos[j] - rel_pos[i1] + search_directions[direct]
                # rj1 = get_equivalent_smallest_vectors(j, i1, supercell, supercell.get_cell(), self._symprec)
                dj1 = np.dot(rj1, supercell.get_cell())
                rj2 = rel_pos[j] - rel_pos[i2] + search_directions[direct]
                # rj2 = get_equivalent_smallest_vectors(j, i2, supercell, supercell.get_cell(), self._symprec)
                dj2 = np.dot(rj2, supercell.get_cell())

                if np.abs(np.dot(dj0, np.cross(dj1, dj2))) < 1e-7: # coplanar
                    scale20 = np.cross(np.cross(d20, d10), d20)
                    s20 = np.dot(dj0, scale20) * scale20 / np.dot(scale20, scale20)
                    scale10 = np.cross(np.cross(d10, d21), d10)
                    s10 = np.dot(dj0, scale10) * scale10 / np.dot(scale10, scale10)
                    scale21 = np.cross(np.cross(d21, d20), d21)
                    s21 = np.dot(dj0, scale21) * scale21 / np.dot(scale21, scale21)

                    ds10d0 = -2 * np.outer(d21, d10) + I * np.dot(d10, d21) + np.outer(d10, d21)
                    ds10d1 = - np.eye(3) * np.dot(d10, d10) + 2 * np.outer(d21, d10) - I * np.dot(d10, d21) - np.outer(d10, d21 - d10)
                    ds10d2 = I * np.dot(d10, d10) - np.outer(d10, d10)
                    ds20d0 = -I * np.dot(d20, d20) - 2 * np.outer(d10, d20) + I * np.dot(d20, d10) + np.outer(d20, d20 + d10)
                    ds20d1 = I * np.dot(d20, d20) - np.outer(d20, d20)
                    ds20d2 = 2 * np.outer(d10, d20) - I * np.dot(d20, d10) - np.outer(d20, d10)
                    ds21d0 = -I * np.dot(d21, d21) + np.outer(d21, d21)
                    ds21d1 = -2 * np.outer(d20, d21) + I * np.dot(d21, d20) + np.outer(d21, d20)
                    ds21d2 = I * np.dot(d21, d21) + 2 * np.outer(d20, d21) - I * np.dot(d21, d20) - np.outer(d21, d21 + d20)


                    Jacobian[direct, j,:, i*3] = dj0 / dist(dj0)
                    Jacobian[direct, i0,:, i*3] = - Jacobian[direct, j,:, i*3]

                    d10dj = scale10 / dist(scale10)
                    d10d0 = (-scale10 + np.dot(dj0-s10, ds10d0)) / dist(scale10)
                    d10d1 = np.dot(dj0-s10, ds10d1) / dist(scale10)
                    d10d2 = np.dot(dj0-s10, ds10d2) / dist(scale10)
                    d20dj = scale20/ dist(scale20)
                    d20d0 = (-scale20 + np.dot(dj0-s20, ds20d0)) / dist(scale20)
                    d20d1 = np.dot(dj0-s20, ds20d1) / dist(scale20)
                    d20d2 = np.dot(dj0-s20, ds20d2) / dist(scale20)
                    d21dj = scale21 / dist(scale21)
                    d21d0 = (-scale21 + np.dot(dj0-s21, ds21d0)) / dist(scale21)
                    d21d1 =  np.dot(dj0-s21, ds21d1) / dist(scale21)
                    d21d2 =  np.dot(dj0-s21, ds21d2) / dist(scale21)

                    if not (np.abs(np.dot(dj0, scale10)) < 1e-8 or np.abs(np.dot(dj0, scale20)) < 1e-8):
                        Jacobian[direct,[i0,i1,i2,j],:,i*3+1] = np.array([d10d0, d10d1, d10d2, d10dj])
                        Jacobian[direct,[i0,i1,i2,j],:,i*3+2] = np.array([d20d0, d20d1, d20d2, d20dj])

                    elif not (np.abs(np.dot(dj0, scale10)) < 1e-8 or np.abs(np.dot(dj0, scale21)) < 1e-8):
                        Jacobian[direct,[i0,i1,i2,j],:,i*3+1] = np.array([d10d0, d10d1, d10d2, d10dj])
                        Jacobian[direct,[i0,i1,i2,j],:,i*3+2] = np.array([d21d0, d21d1, d21d2, d21dj])
                    elif not (np.abs(np.dot(dj0, scale21)) < 1e-8 or np.abs(np.dot(dj0, scale20)) < 1e-8):
                        Jacobian[direct,[i0,i1,i2,j],:,i*3+1] = np.array([d21d0, d21d1, d21d2, d21dj])
                        Jacobian[direct,[i0,i1,i2,j],:,i*3+2] = np.array([d20d0, d20d1, d20d2, d20dj])
                    else:
                        print "Error"
                else:
                    Jacobian[direct,j, :, i*3] = dj0 / dist(dj0)
                    Jacobian[direct,i0,:,i*3] = - Jacobian[direct,j, :, i*3]
                    Jacobian[direct,j,:,i*3+1] = dj1 / dist(dj1)
                    Jacobian[direct,i1,:,i*3+1] = - Jacobian[direct,j,:,i*3+1]
                    Jacobian[direct,j,:,i*3+2] = dj2 / dist(dj2)
                    Jacobian[direct,i2,:,i*3+2] = - Jacobian[direct,j,:,i*3+2]

        # unit_atoms = np.unique(supercell.get_supercell_to_unitcell_map())
        # unit_coords = np.concatenate([i * 3 + np.arange(3) for i in unit_atoms])

        # J0 = Jacobian[unique_atoms].reshape(-1, natom * 3 - 6)
        # J = Jacobian.reshape(-1, natom * 3 - 6)
        # non_zero_pos0 = np.where(np.abs(J0)>1e-8)
        # non_zero_pos = np.where(np.abs(J)>1e-8)
        # plt.matshow(J)
        # plt.axis('off')
        # plt.savefig('Jacobian.pdf')
        # J0_sparse = coo_matrix((J0[non_zero_pos0], non_zero_pos0), shape = J0.shape)
        # J_sparse = coo_matrix((J[non_zero_pos], non_zero_pos), shape = J.shape)

        # eta0_sparse = scipy.sparse.kron(J0_sparse, J_sparse)
        # eta_sparse = scipy.sparse.kron(J0_sparse, J_sparse)
        # eta = np.kron(J0, J).reshape(len(unique_atoms), natom, -1)
        eta = np.zeros(((len(unique_atoms), natom, 9, (natom*3-6)**2)), dtype='double')
        # eta_reduced = []
        # for ele in self._ifc2_ele:
        #     p1, p2 = self._pairs_reduced[ele / 9]
        #     i = (ele % 9) / 3
        #     j = (ele % 9) % 3
        #     vecs, indices = get_equivalent_smallest_vectors(p2, p1, supercell, supercell.get_cell(), self._symprec, is_return_index=True)
        #     eta_reduced.append(np.kron(Jacobian[0, p1, i], Jacobian[indices[0], p2, j]))
        # eta_reduced = np.array(eta_reduced, dtype='double')
        # for i, ip in enumerate(unique_atoms):
        #     for j in range(natom):
        #         if len(get_equivalent_smallest_vectors(j, ip, supercell, supercell.get_cell(), self._symprec)) > 1:
        #             continue
        #         coeff = np.dot(self._coeff2[ip,j], self._ifc2_trans[self._ifc2_map[ip, j]])
        #         eta[i, j] = np.dot(coeff, eta_reduced)
        for i, ip in enumerate(unique_atoms):
            for j in range(natom):
                # if len(get_equivalent_smallest_vectors(j, ip, supercell, supercell.get_cell(), self._symprec)) > 1:
                #     continue
                # coeff = np.dot(self._coeff2[ip,j], self._ifc2_trans[self._ifc2_map[ip, j]])
                vecs, indices = get_equivalent_smallest_vectors(j, ip, supercell, supercell.get_cell(), self._symprec, is_return_index=True)
                if len(indices) > 1:
                    continue
                for k in range(9):
                    eta[i, j, k] = np.kron(Jacobian[0, ip, k / 3], Jacobian[indices[0], j, k % 3])


        eta = eta.reshape(len(unique_atoms)* natom* 9, -1)
        non_zero_eta = np.where(np.abs(eta)>1e-10)
        eta_sparse = coo_matrix((eta[non_zero_eta], non_zero_eta), shape = eta.shape)
        fc = self._fc2_read[unique_atoms].flatten()
        # fc0 = force_constants[unit_atoms].swapaxes(1,2).flatten()
        force_constants_internal = lsqr(eta_sparse, fc)
        print "lst error: %10.5e" %force_constants_internal[3]
        force_constants_new = eta_sparse.dot(force_constants_internal[0])
        force_constants_new = force_constants_new.reshape((len(unique_atoms), natom, 3, 3))

        self._fc2 = np.zeros_like(self._fc2_read)
        s2u_map = self._supercell.get_supercell_to_unitcell_map()
        scaled_positions = self._supercell.get_scaled_positions()
        for i in range(natom):
            ip = s2u_map[i]
            diff = scaled_positions[i] - scaled_positions[ip]
            for j in range(natom):
                disp = scaled_positions[j] - diff - scaled_positions
                jp = np.where(np.all(np.abs(disp - np.rint(disp)) < self._symprec, axis=-1))[0][0]
                self._fc2[i, j] = force_constants_new[np.where(unique_atoms == ip)[0][0], jp]
        write_fc2_hdf5(self._fc2, filename='fc2-tuned.hdf5')
        plt.figure()
        plt.scatter(fc, force_constants_new.flatten(), color = 'red', s=4)
        plt.plot(np.array([fc.min(), fc.max()]), np.array([fc.min(), fc.max()]), color='blue')
        plt.savefig('fc2_compare.pdf')
        # plt.figure()


    def set_fc3_read(self, fc3_filename):
        self._fc3_read = read_fc3_from_hdf5(filename=fc3_filename)

    def set_first_independents(self):
        sym = self._symmetry
        independents = {}
        independents['atoms'] = sym.get_independent_atoms() # independent atoms
        independents['natoms'] = len(independents['atoms']) # number of independent atoms
        sym_operations = sym.get_symmetry_operations()
        independents['rotations'] = sym_operations['rotations']
        independents['translations'] = sym_operations['translations']
        independents['noperations'] = len(sym_operations['rotations'])
        # rotations and translations forms all the operations
        independents['mappings'] = sym.get_map_atoms()
        independents['mapping_operations'] = sym.get_map_operations()
        self._ind1 = independents

    def set_second_independents(self, pair_included=None):
        "pair_included is a natom_super * natom_super matrix determing whether fc2 between any two atoms in the supercell is included"
        symprec = self._symmetry.get_symmetry_tolerance()
        positions = self._supercell.get_scaled_positions()
        indep = {}
        symmetry = self._symmetry
        if self._ind1 == None:
            self.set_first_independents()

        site_syms_list = []
        for atom1 in self._ind1['atoms']:
            site_syms_list.append(symmetry.get_site_symmetry(atom1))
        indep['noperations'] = np.array([len(site_sym) for site_sym in site_syms_list], dtype="intc")
        max_size = np.max(indep['noperations'])
        site_syms = np.zeros((self._ind1['natoms'], max_size, 3, 3), dtype=np.float)
        for i in range(self._ind1['natoms']):
            site_syms[i,:indep['noperations'][i]] = np.array(site_syms_list[i])
        indep['rotations'] = site_syms
        indep['translations'] = None
        mappings = np.zeros((self._ind1['natoms'], self._num_atom), dtype="intc")
        mapping_operations = np.zeros((self._ind1['natoms'], self._num_atom), dtype="intc")
        for i, atom1 in enumerate(self._ind1['atoms']):
            mappings[i], mapping_operations[i] = get_atom_mapping(atom1,
                                                                  np.array(site_syms_list[i]),
                                                                  positions,
                                                                  symprec=symprec,
                                                                  is_return_opes=True)
        indep['mappings'] = mappings
        indep['mapping_operations'] = mapping_operations
        indep['natoms'] = np.array([len(np.unique(mappings[i])) for i in range(self._ind1['natoms'])], dtype="intc")
        ind_atom2 = np.zeros((self._ind1['natoms'], np.max(indep['natoms'])), dtype="intc")
        included_atom2 = np.ones_like(ind_atom2, dtype="bool")
        for i, atom1 in enumerate(self._ind1['atoms']):
            ind_atom2[i, :indep['natoms'][i]] = np.unique(mappings[i])
            if pair_included is not None:
                included_atom2[i,:indep['natoms'][i]] = pair_included[atom1, np.unique(mappings[i])]
        indep['atoms'] = ind_atom2
        indep['included'] = included_atom2
        self._ind2 = indep
        pairs = []
        for i, atom1 in enumerate(self._ind1['atoms']):
            for j, atom2 in enumerate(self._ind2['atoms'][i, :self._ind2['natoms'][i]]):
                pairs.append((atom1, atom2))
        self._pairs = pairs

    def set_pair_reduced_included(self, pair_inclusion=None):
        pairs_included = np.ones(len(self._pairs_reduced), dtype=bool)
        if pair_inclusion is not  None:
            for i, (atom1, atom2) in enumerate(self._pairs_reduced):
                pairs_included[i] = pair_inclusion[atom1, atom2]
        self._pairs_included = pairs_included

    def get_irreducible_fc2s_with_permute(self):
        ind1 = self._ind1
        ind2 = self._ind2
        self._pair_mappings, self._pair_transforms = \
            get_pairs_with_permute(self._pairs,
                                   self._supercell.get_cell().T,
                                   self._supercell.get_scaled_positions(),
                                   ind1['rotations'],
                                   ind1['translations'],
                                   ind1['atoms'],
                                   ind1['mappings'],
                                   ind1['mapping_operations'],
                                   ind2['rotations'],
                                   ind2['mappings'],
                                   ind2['mapping_operations'])
        self._pairs_reduced = [self._pairs[i] for i in np.unique(self._pair_mappings)]

    def get_fc2_coefficients(self):
        ind1 = self._ind1
        ind2 = self._ind2
        self._coeff2, self._ifc2_map =\
            get_fc2_coefficients(self._pairs,
                                 self._pair_mappings,
                                 self._pair_transforms,
                                 self._supercell.get_cell().T,
                                 self._supercell.get_scaled_positions(),
                                 ind1['rotations'],
                                 ind1['translations'],
                                 ind1['atoms'],
                                 ind1['mappings'],
                                 ind1['mapping_operations'],
                                 ind2['rotations'],
                                 ind2['mappings'],
                                 ind2['mapping_operations'])

    def get_irreducible_fc2_components_with_spg(self):
        ind1 = self._ind1
        ind2 = self._ind2
        self._ifc2_ele, self._ifc2_trans = \
            get_fc2_spg_invariance(np.array(self._pairs_reduced)[np.where(self._pairs_included)],
                                   self._supercell.get_scaled_positions(),
                                   ind1['rotations'],
                                   ind1['translations'],
                                   ind1['mappings'],
                                   ind2['rotations'],
                                   ind2['noperations'],
                                   ind2['mappings'],
                                   self._supercell.get_cell().T,
                                   self._symmetry.get_symmetry_tolerance())

    def get_fc2_translational_invariance(self):
        if self._fc2_read is not None:
            fc2_reduced_pair = np.array([self._fc2_read[pair] for pair in self._pairs_reduced])
            fc2_irr_orig = fc2_reduced_pair.flatten()[self._ifc2_ele]
        irreducible_tmp, transform_tmp = \
            get_fc2_translational_invariance(self._supercell,
                                             self._ifc2_trans,
                                             self._coeff2,
                                             self._ifc2_map,
                                             precesion=self._precision)
        self._ifc2_ele = self._ifc2_ele[irreducible_tmp]
        self._ifc2_trans = np.dot(self._ifc2_trans, transform_tmp)
        #checking the results of gaussian elimination

        if self._fc2_read is not None:
            fc2_irr_new = fc2_irr_orig[irreducible_tmp]
            error = fc2_irr_orig - np.dot(transform_tmp, fc2_irr_new)
            if np.sqrt(np.sum(error ** 2)) > 1:
                print "##################################################################################"
                print "Largest translational drift from the original irreducible fc2:%15.5e" %np.sqrt(np.sum(error ** 2))
                print "Warning! The translational invariance creates somewhat too strict constraints"
                print "Maybe you want to lower the precesion standard to tolerate more noise"
                print "##################################################################################"


    def get_fc2_rotational_invariance(self):
        if self._fc2_read is not None:
            fc2_reduced_pair = np.array([self._fc2_read[pair] for pair in self._pairs_reduced])
            fc2_irr_orig = fc2_reduced_pair.flatten()[self._ifc2_ele]

        irreducible_tmp, transform_tmp = \
            get_fc2_rotational_invariance(self._supercell,
                                          self._ifc2_trans,
                                          self._coeff2,
                                          self._ifc2_map,
                                          symprec=self._symprec,
                                          precision=self._precision)
        self._ifc2_ele = self._ifc2_ele[irreducible_tmp]
        self._ifc2_trans = np.dot(self._ifc2_trans, transform_tmp)
        if self._fc2_read is not None:
            fc2_irr_new = fc2_irr_orig[irreducible_tmp]
            error = fc2_irr_orig - np.dot(transform_tmp, fc2_irr_new)
            if np.sqrt(np.sum(error ** 2)) > 1:
                print "##################################################################################"
                print "Largest rotational drift from the original irreducible fc2:%15.5e" %np.sqrt(np.sum(error ** 2))
                print "Warning! This rotational invariance creates somewhat too strict constraints"
                print "Maybe you want to lower the precesion standard to tolerate more noise"
                print "##################################################################################"

    def set_expand_fc2_conditions(self):
        irreducible_tmp, transform_tmp = \
            expand_fc2_conditions(self._supercell,
                                  self._ifc2_trans,
                                  self._coeff2,
                                  self._ifc2_map,
                                  symprec=self._symprec,
                                  precesion=self._precision)
        self._ifc2_ele = self._ifc2_ele[irreducible_tmp]
        self._ifc2_trans = np.dot(self._ifc2_trans, transform_tmp)
        print "Maximum interaction distance further reduces independent fc2 components to %d" %len(self._ifc2_ele)

    def set_fc2_irreducible_elements(self, is_trans_inv=False, is_rot_inv=False):
        self.set_first_independents()
        print "Under the system symmetry"
        print "Number of first independent atoms: %4d" % self._ind1['natoms']
        self.set_second_independents(pair_included=self._cutoff.get_pair_inclusion())
        print "Number of second independent atoms" + " %4d" * len(self._ind2['natoms']) % tuple(self._ind2['natoms'])
        self.get_irreducible_fc2s_with_permute()
        print "Permutation symmetry reduces number of irreducible pairs from %4d to %4d"\
              %(self._ind2['natoms'].sum(), len(self._pairs_reduced))
        if (self._cutoff is not None) and (self._cutoff.get_cutoff_pair() is not None):
            pair_inclusion = self._cutoff.get_pair_inclusion()
            self.set_pair_reduced_included(pair_inclusion)
            print "The artificial cutoff reduces number of irreducible pairs from %4d to %4d"\
                  %(len(self._pairs_reduced), np.sum(self._pairs_included))
        else:
            self.set_pair_reduced_included()
        print "Calculating transformation coefficients..."
        print "Number of independent fc2 components: %d" %(np.sum(self._pairs_included)*9)
        self.get_irreducible_fc2_components_with_spg()
        self.get_fc2_coefficients()

        print "Point group invariance reduces independent fc2 components to %d" %(len(self._ifc2_ele))
        sys.stdout.flush()
        if self._cutoff is None or self._cutoff._cut_pair is None:
            if is_trans_inv:
                self.get_fc2_translational_invariance()
                print "Translational invariance further reduces independent fc2 components to %d" %len(self._ifc2_ele)
            if is_rot_inv:
                self.get_fc2_rotational_invariance()
                print "Rotational invariance further reduces independent fc2 components to %d" %len(self._ifc2_ele)
        print "Independent fc2 components calculation completed"
        if DEBUG:
            from mdfc.file_IO import read_fc2_from_hdf5
            fc2 = read_fc2_from_hdf5("fc2.hdf5")
            fc2_reduced = np.array([fc2[pai] for pai in self._pairs_reduced])
            fc2p = fc2_reduced.flatten()[self._ifc2_ele]
            pp = np.einsum('ijkl, ijl-> ijk', self._coeff2, fc2_reduced[self._ifc2_map].reshape(self._num_atom, self._num_atom, 9)).reshape(self._num_atom, self._num_atom, 3, 3)
        sys.stdout.flush()

    def tune_fc2(self, is_minimize_relative_error=False, log_level=1):
        len_element = len(self._ifc2_ele)
        transform = np.zeros((self._num_atom, self._num_atom, 9, len_element), dtype='double')
        for i, j in np.ndindex((self._num_atom, self._num_atom)):
            transform[i,j] = np.dot(self._coeff2[i,j], self._ifc2_trans[self._ifc2_map[i, j]])
        transform2 = transform.reshape(-1, len_element)
        fc2_read_flatten = self._fc2_read.flatten()
        if is_minimize_relative_error:
            fc_scale = np.zeros_like(self._fc2_read)
            transform2 = np.zeros_like(transform)
            for i, j in np.ndindex((self._num_atom, self._num_atom)):
                fc_factor = np.linalg.norm(self._fc2_read[i,j])
                transform2[i,j] = transform[i,j] / fc_factor
                fc_scale[i,j] = self._fc2_read[i,j] / fc_factor
            transform2 = transform2.reshape(-1, len_element)
            fc2_read_flatten = fc_scale.flatten()
        try:
            import scipy
            non_zero = np.where(np.abs(transform2) > 1e-8)
            transform_sparse = scipy.sparse.coo_matrix((transform2[non_zero], non_zero), shape=transform2.shape)
            lsqr_results = scipy.sparse.linalg.lsqr(transform_sparse, fc2_read_flatten)
            fc_irred = lsqr_results[0]
            error = lsqr_results[3]
            fc_tuned = np.dot(transform, fc_irred)
        except ImportError:
            transform_pinv = np.linalg.pinv(transform2)
            fc_irred = np.dot(transform_pinv, fc2_read_flatten)
            fc_tuned = np.dot(transform, fc_irred)
            error = np.sqrt(np.sum((fc2_read_flatten - fc_tuned.flatten())**2))
        self._fc2 = fc_tuned.reshape(self._num_atom, self._num_atom, 3, 3)
        print "FC2 tunning process using the least-square method has completed"
        print "    with least square error: %f (eV/A^2)" %error
        if log_level == 2:
            print "The comparison between original and the tuned force constants is plot and saved to f2-tune_compare.pdf"
            plt.figure()
            plt.scatter(self._fc2_read.flatten(), fc_tuned, color='red', s=3)
            plt.plot(np.array([self._fc2_read.min(), self._fc2_read.max()]),
                     np.array([self._fc2_read.min(), self._fc2_read.max()]), color='blue')
            threshold = 10 ** np.rint(np.log10(np.abs(self._fc2_read).max() / 1e3))
            plt.yscale('symlog', linthreshy=threshold)
            plt.xscale('symlog', linthreshx=threshold)
            plt.xlabel("Original fc2 (eV/A^2)")
            plt.ylabel("Tuned fc2 (eV/A^2)")
            plt.savefig("fc_tune_compare.pdf")
        write_fc2_hdf5(self._fc2, filename='fc2-tuned.hdf5')

    def tune_fc3(self):
        len_element = len(self._ifc3_ele)
        first_atoms = np.unique(self._supercell.get_supercell_to_unitcell_map())

        transform = np.zeros((len(first_atoms), self._num_atom,self._num_atom, 27, len_element), dtype='double')
        for first, i in enumerate(first_atoms):
            for j, k in np.ndindex((self._num_atom, self._num_atom)):
                transform[first, j, k] = np.dot(self._coeff3[i,j, k], self._ifc3_trans[self._ifc3_map[i, j, k]])
        transform2 = transform.reshape(-1, len_element)
        fc3_read = self._fc3_read[first_atoms].flatten()
        try:
            raise ImportError
            import scipy
            non_zero = np.where(np.abs(transform2) > 1e-8)
            transform_sparse = scipy.sparse.coo_matrix((transform2[non_zero], non_zero), shape=transform2.shape)
            lsqr_results = scipy.sparse.linalg.lsqr(transform_sparse, fc3_read)
            self._fc3_irred = lsqr_results[0]
            fc_tuned = np.dot(transform2, self._fc3_irred)
            error = lsqr_results[3]
        except ImportError:
            transform_pinv = np.linalg.pinv(transform2)
            self._fc3_irred = np.dot(transform_pinv, fc3_read)
            fc_tuned = np.dot(transform2, self._fc3_irred)
            error = np.sqrt(np.sum((fc_tuned - fc3_read)**2))
        print "FC3 tunning process using the least-square method has completed"
        print "    with least square error: %f (eV/A^3)" %error

        if DEBUG:
            plt.figure()
            plt.scatter(fc3_read.flatten(), fc_tuned.flatten(), color='red', s=3)
            plt.plot(np.array([fc3_read.min(), fc3_read.max()]),
                     np.array([fc3_read.min(), fc3_read.max()]), color='blue')
            plt.savefig("fc3_tune_compare.pdf")
        self.distribute_fc3()
        write_fc3_hdf5(self._fc3, filename='fc3-tuned.hdf5')

    def set_fc3_irreducible_components(self, is_trans_inv = False, is_rot_inv = False):
        self.set_third_independents()
        print "Number of 3rd IFC: %d" %(27 * len(self._triplets))
        self.get_irreducible_fc3s_with_permute()
        print "Permutation reduces 3rd IFC to %d"%(27 * len(self._triplets_reduced))
        self.get_irreducible_fc3_components_with_spg()
        print "spg invariance reduces 3rd IFC to %d"%(len(self._ifc3_ele))
        print "Calculating fc3 coefficient..."
        self.get_fc3_coefficients()
        if DEBUG:
            fc3_read = self._fc3_read
            fc3_reduced_triplets = np.double([fc3_read[index] for index in self._triplets_reduced])
            for i, j, k in np.ndindex((self._num_atom, self._num_atom, self._num_atom)):
                fc3_triplet = np.dot(self._coeff3[i,j,k], fc3_reduced_triplets[self._ifc3_map[i,j,k]].flatten())
                fc3_orig = fc3_read[i,j,k]
                diff = fc3_orig.flatten() - fc3_triplet
                if (np.abs(diff) > 1e-1).any():
                    print np.abs(diff).max()
        if is_trans_inv:
            self.get_fc3_translational_invariance()
            print "translational invariance reduces 3rd IFC to %d"%(len(self._ifc3_ele))
        if DEBUG:
            fc3_read = self._fc3_read
            fc3_reduced_triplets = np.double([fc3_read[index] for index in self._triplets_reduced])
            fc3_irr = fc3_reduced_triplets.flatten()[self._ifc3_ele]
            fc3_reduced_triplets2 = np.dot(self._ifc3_trans, fc3_irr)
        if is_rot_inv and self._fc2 is not None:
            self.get_fc3_rotational_invariance(self._fc2)
            print "rotational invariance reduces 3rd IFC to %d"%(len(self._ifc3_ele))
        self._num_irred_fc3 = len(self._ifc3_ele)
        sys.stdout.flush()

    def distribute_fc3(self):
        coeff = self._coeff3
        ifcmap = self._ifc3_map
        trans = self._ifc3_trans
        num_irred = trans.shape[-1]
        assert self._fc3_irred is not None
        # distribute all the fc3s
        print "Distributing fc3..."
        fc3 = np.zeros((self._num_atom, self._num_atom, self._num_atom, 3,3,3), dtype="double")
        for atom1 in np.arange(self._num_atom):
            for atom2 in np.arange(self._num_atom):
                for atom3 in np.arange(self._num_atom):
                    num_triplet = ifcmap[atom1, atom2, atom3]
                    coeff_temp = np.dot(coeff[atom1, atom2, atom3], trans[num_triplet]).reshape(3,3,3, num_irred)
                    fc3[atom1, atom2, atom3] = np.dot(coeff_temp, self._fc3_irred).reshape(3,3,3)
        self._fc3 = fc3


    @timeit
    def set_third_independents(self):
        symprec = self._symmetry.get_symmetry_tolerance()
        positions = self._supercell.get_scaled_positions()
        if self._ind1 == None:
            self.set_first_independents()
        ind1 = self._ind1
        if self._ind2 == None:
            self.set_second_independents(self._cutoff.get_pair_inclusion())
        ind2 = self._ind2
        ind3 = {}
        nind_atoms = np.zeros_like(ind2['atoms'])
        ind_mappings = np.zeros(ind2['atoms'].shape + (self._num_atom,), dtype="intc")
        ind_mapping_opes = np.zeros_like(ind_mappings)
        all_bond_syms_list = []
        for i, atom1 in enumerate(ind1['atoms']):
            site_sym = ind2['rotations'][i]
            bond_syms = []
            for j, atom2 in enumerate(ind2['atoms'][i]):
                reduced_bond_sym = get_bond_symmetry(
                    site_sym,
                    positions,
                    atom1,
                    atom2,
                    symprec)
                bond_syms.append(reduced_bond_sym)
                atom_mappings, mapping_operations = get_atom_mapping(atom2, reduced_bond_sym, positions, symprec, True)
                ind_mappings[i,j] = atom_mappings
                ind_mapping_opes[i,j] = mapping_operations
                nind_atoms[i,j] = len(np.unique(atom_mappings))
            all_bond_syms_list.append(bond_syms)
        bond_sym_sizes = [[len(bond2) for bond2 in bond1] for bond1 in all_bond_syms_list]
        max_bond_sym = np.max([np.max(bond) for bond in bond_sym_sizes])
        all_bond_syms = np.zeros(ind2['atoms'].shape + (max_bond_sym, 3, 3), dtype="intc")
        nsym = np.zeros(ind2['atoms'].shape, dtype="intc")
        ind_atoms = np.zeros(ind2['atoms'].shape + (np.max(nind_atoms),), dtype="intc")
        for i, atom1 in enumerate(ind1['atoms']):
            for j, atom2 in enumerate(ind2['atoms'][i]):
                length = bond_sym_sizes[i][j]
                all_bond_syms[i,j, :length] = all_bond_syms_list[i][j]
                nsym[i,j] = length
                ind_atoms[i,j, :nind_atoms[i,j]] = np.unique(ind_mappings[i,j])
        ind3['mappings'] = ind_mappings
        ind3['mapping_operations'] = ind_mapping_opes
        ind3['noperations'] = nsym
        ind3['rotations'] = all_bond_syms
        ind3['translations'] = None
        ind3['atoms'] = ind_atoms
        ind3['natoms'] = nind_atoms
        self._ind3 = ind3
        triplets = []
        for i, atom1 in enumerate(self._ind1['atoms']):
            for j, atom2 in enumerate(self._ind2['atoms'][i, :self._ind2['natoms'][i]]):
                for k, atom3 in enumerate(self._ind3['atoms'][i, j, :self._ind3['natoms'][i,j]]):
                    triplets.append((atom1, atom2, atom3))
        self._triplets = triplets


    @timeit
    def get_irreducible_fc3s_with_permute(self):
        ind1 = self._ind1
        ind2 = self._ind2
        ind3 = self._ind3
        self._triplet_mappings, self._triplet_transforms = \
            get_irreducible_triplets_with_permute(self._triplets,
                                                  self._positions,
                                                  ind1['rotations'],
                                                  ind1['translations'],
                                                  ind1['mappings'],
                                                  ind1['mapping_operations'],
                                                  ind2['rotations'],
                                                  ind2['noperations'],
                                                  ind2['mappings'],
                                                  ind2['mapping_operations'],
                                                  ind3['rotations'],
                                                  ind3['noperations'],
                                                  ind3['mappings'],
                                                  ind3['mapping_operations'],
                                                  self._lattice,
                                                  self._symprec)
        self._triplets_reduced = [self._triplets[t] for t in np.unique(self._triplet_mappings)]

        # self._triplets_reduced = [self._triplets[i] for i in np.unique(self._triplet_mappings)]

    @timeit
    def get_fc3_coefficients(self, lang="C"):
        ind1 = self._ind1
        ind2 = self._ind2
        ind3 = self._ind3
        if lang=="py":
            self._coeff3, self._ifc3_map =\
                get_fc3_coefficients(self._triplets,
                                     self._triplet_mappings,
                                     self._triplet_transforms,
                                     self._lattice,
                                     self._positions,
                                     ind1['rotations'],
                                     ind1['translations'],
                                     ind1['mappings'],
                                     ind1['mapping_operations'],
                                     ind2['rotations'],
                                     ind2['mappings'],
                                     ind2['mapping_operations'],
                                     ind3['rotations'],
                                     ind3['mappings'],
                                     ind3['mapping_operations'],
                                     self._symprec)
        else:
            import _mdfc
            self._coeff3 = np.zeros((self._num_atom, self._num_atom, self._num_atom, 27, 27), dtype="double")
            self._ifc3_map = np.zeros((self._num_atom, self._num_atom, self._num_atom), dtype="intc")
            _mdfc.get_fc3_coefficients(self._coeff3,
                                         self._ifc3_map,
                                         np.array(self._triplets).astype("intc"),
                                         self._triplet_mappings.copy().astype("intc"),
                                         self._triplet_transforms.copy().astype("double"),
                                         self._lattice.copy().astype("double"),
                                         self._positions.copy().astype("double"),
                                         ind1['rotations'].astype("intc"),
                                         ind1['translations'].astype("double"),
                                         ind1['mappings'].astype("intc"),
                                         ind1['mapping_operations'].astype("intc"),
                                         ind2['rotations'].astype("intc"),
                                         ind2['mappings'].astype("intc"),
                                         ind2['mapping_operations'].astype("intc"),
                                         ind3['rotations'].astype("intc"),
                                         ind3['mappings'].astype("intc"),
                                         ind3['mapping_operations'].astype("intc"),
                                         self._symprec)


    @timeit
    def get_irreducible_fc3_components_with_spg(self, lang="C"):
        ind1 = self._ind1
        ind2 = self._ind2
        ind3 = self._ind3
        if lang == "py":
            self._ifc3_ele, self._ifc3_trans = \
                get_fc3_spg_invariance(self._triplets_reduced,
                                       self._positions,
                                       ind1['rotations'],
                                       ind1['translations'],
                                       ind1['mappings'],
                                       ind2['rotations'],
                                       ind2['noperations'],
                                       ind2['mappings'],
                                       ind3['rotations'],
                                       ind3['noperations'],
                                       ind3['mappings'],
                                       self._lattice,
                                       self._symprec)
        else:
            import _mdfc
            self._ifc3_ele, self._ifc3_trans = \
            _mdfc.get_fc3_spg_invariance(np.array(self._triplets_reduced, dtype="intc"),
                                       self._positions.copy(),
                                       ind1['rotations'].astype("intc"),
                                       ind1['translations'].copy(),
                                       ind1['mappings'].astype("intc"),
                                       ind2['rotations'].astype("intc"),
                                       ind2['noperations'].astype("intc"),
                                       ind2['mappings'].astype("intc"),
                                       ind3['rotations'].astype("intc"),
                                       ind3['noperations'].astype("intc"),
                                       ind3['mappings'].astype("intc"),
                                       self._lattice.copy(),
                                       self._symprec)
        if DEBUG:
            fc3_read = self._fc3_read
            fc3_reduced_triplets = np.double([fc3_read[index] for index in self._triplets_reduced])
            fc3_reduced =  fc3_reduced_triplets.flatten()[self._ifc3_ele]
            fc3_reduced_triplets2 = np.dot(self._ifc3_trans, fc3_reduced)
            diff = fc3_reduced_triplets2 - fc3_reduced_triplets.reshape(-1, 27)
            print np.abs(diff).max()

    def get_fc3_translational_invariance(self):
        if self._fc3_read is not None:
            fc3_reduced_pair = np.array([self._fc3_read[triplet] for triplet in self._triplets_reduced])
            fc3_irr_orig = fc3_reduced_pair.flatten()[self._ifc3_ele]
        irreducible_tmp, transform_tmp = \
            get_fc3_translational_invariance(self._supercell,
                                             self._ifc3_trans,
                                             self._coeff3,
                                             self._ifc3_map,
                                             self._precision)
        self._ifc3_ele = self._ifc3_ele[irreducible_tmp]
        self._ifc3_trans = np.dot(self._ifc3_trans, transform_tmp)
        if self._fc3_read is not None:
            fc3_irr_new = fc3_irr_orig[irreducible_tmp]
            error = fc3_irr_orig - np.dot(transform_tmp, fc3_irr_new)
            if np.sqrt(np.sum(error ** 2)) > 1:
                print "##################################################################################"
                print "Largest rotational drift from the original irreducible fc2:%15.5e" %np.sqrt(np.sum(error ** 2))
                print "Warning! This rotational invariance creates somewhat too strict constraints"
                print "Maybe you want to lower the precesion standard to tolerate more noise"
                print "##################################################################################"

    def get_fc3_rotational_invariance(self, fc2):
        if self._fc3_read is not None:
            fc3_reduced_pair = np.array([self._fc3_read[triplet] for triplet in self._triplets_reduced])
            fc3_irr_orig = fc3_reduced_pair.flatten()[self._ifc3_ele]
        irreducible_tmp, transform_tmp =\
            get_fc3_rotational_invariance(fc2,
                                          self._supercell,
                                          self._ifc3_trans,
                                          self._coeff3,
                                          self._ifc3_map,
                                          self._symprec,
                                          precision=self._precision)
        self._ifc3_ele = self._ifc3_ele[irreducible_tmp]
        self._ifc3_trans = np.dot(self._ifc3_trans, transform_tmp)
        if self._fc3_read is not None:
            fc3_irr_new = fc3_irr_orig[irreducible_tmp]
            error = fc3_irr_orig - np.dot(transform_tmp, fc3_irr_new)
            if np.sqrt(np.sum(error ** 2)) > 1:
                print "##################################################################################"
                print "Largest rotational drift from the original irreducible fc2:%15.5e" %np.sqrt(np.sum(error ** 2))
                print "Warning! This rotational invariance creates somewhat too strict constraints"
                print "Maybe you want to lower the precesion standard to tolerate more noise"
                print "##################################################################################"

def set_translational_invariance(force_constants):
    """
    Translational invariance is imposed.  This is quite simple
    implementation, which is just taking sum of the force constants in
    an axis and an atom index. The sum has to be zero due to the
    translational invariance. If the sum is not zero, this error is
    uniformly subtracted from force constants.
    """
    set_translational_invariance_per_index(force_constants, index=0)
    set_translational_invariance_per_index(force_constants, index=1)

def set_translational_invariance_per_index(force_constants, index=0):
    if index == 0:
        for i in range(force_constants.shape[1]):
            for j in range(force_constants.shape[2]):
                for k in range(force_constants.shape[3]):
                    force_constants[:, i, j, k] -= np.sum(
                        force_constants[:, i, j, k]) / force_constants.shape[0]
    elif index == 1:
        for i in range(force_constants.shape[0]):
            for j in range(force_constants.shape[2]):
                for k in range(force_constants.shape[3]):
                    force_constants[i, :, j, k] -= np.sum(
                        force_constants[i, :, j, k]) / force_constants.shape[1]


def set_permutation_symmetry(force_constants):
    """
    Phi_ij_ab = Phi_ji_ba

    i, j: atom index
    a, b: Cartesian axis index

    This is not necessary for harmonic phonon calculation because this
    condition is imposed when making dynamical matrix Hermite in
    dynamical_matrix.py.
    """
    fc_copy = force_constants.copy()
    for i in range(force_constants.shape[0]):
        for j in range(force_constants.shape[1]):
            force_constants[i, j] = (force_constants[i, j] +
                                     fc_copy[j, i].T) / 2

def show_rotational_invariance(force_constants,
                          supercell,
                          primitive,
                          symprec=1e-5):
    """
    *** Under development ***
    Just show how force constant is close to the condition of rotational invariance,
    """
    print "Check rotational invariance ..."

    fc = force_constants
    p2s = primitive.get_primitive_to_supercell_map()

    abc = "xyz"

    for pi, p in enumerate(p2s):
        for i in range(3):
            mat = np.zeros((3, 3), dtype='double')
            for s in range(supercell.get_number_of_atoms()):
                vecs = np.array(get_equivalent_smallest_vectors(
                        s, p, supercell, primitive.get_cell(), symprec))
                m = len(vecs)
                v = np.dot(vecs[:,:].sum(axis=0) / m, primitive.get_cell())
                for j in range(3):
                    for k in range(3):
                        mat[j, k] += (fc[p, s, i, j] * v[k] -
                                      fc[p, s, i, k] * v[j])

            print "Atom %d %s" % (p+1, abc[i])
            for vec in mat:
                print "%10.5f %10.5f %10.5f" % tuple(vec)

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

if __name__=="__main__":


    from phonopy.interface import vasp
    from phonopy.structure.symmetry import Symmetry
    from phonopy.structure.cells import get_supercell
    unitcell = vasp.read_vasp("POSCAR")
    supercell=get_supercell(unitcell,2 * np.eye(3, dtype="intc"))
    symmetry  = Symmetry(supercell)
    fc = ForceConstants(supercell, symmetry)
    fc.set_first_independents()
    fc.set_second_independents()
    fc.get_irreducible_fc2s_with_permute()
    fc.get_fc2_coefficients()
    fc.get_irreducible_fc2_components_with_spg()
    fc.get_fc2_translational_invariance()
    fc.get_fc2_rotational_invariance()
    fc.set_third_independents()
    fc.get_irreducible_fc3s_with_permute()
    fc.get_irreducible_fc3_components_with_spg()
    fc.get_fc3_coefficients()
    fc.get_fc3_translational_invariance()
    print