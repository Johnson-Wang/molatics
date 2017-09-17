import numpy as np
import sys
import scipy
from itertools import permutations
from scipy.sparse import dok_matrix, coo_matrix, csr_matrix
from fc2 import get_fc2_coefficients, get_fc2_spg_invariance,\
    get_fc2_translational_invariance, get_fc2_rotational_invariance, get_trim_fc2, get_index_of_atom

from fc3 import get_fc3_coefficients, get_fc3_spg_invariance,\
    get_fc3_translational_invariance, get_fc3_rotational_invariance, get_trim_fc3
from file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5, write_fc2_hdf5, write_fc3_hdf5
from fcmath import mat_dot_product, mat_dense_to_sparse
from realmd.information import timeit
import matplotlib.pyplot as plt
from realmd.memory_profiler import profile
DEBUG = False

search_directions = np.array(list(np.ndindex((3,3,3)))) - 1

def check_descrepancy(fc, fc_orig, info='', threshold=1):
    error = np.sqrt(np.sum((fc_orig - fc) ** 2))
    if error > threshold:
        print "##################################################################################"
        print "Largest %s drift from the original irreducible fc2:%15.5e" %(info, error)
        print "Warning! This %s invariance creates somewhat too strict constraints" %info
        print "Maybe you want to lower the precision standard to tolerate more noise"
        print "##################################################################################"

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

from mdfc.symmetry import Symmetry
class ForceConstants():
    def __init__(self, supercell, primitive, symmetry, is_disperse=False, cutoff=None, precision=1e-8):
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
        self._is_disperse=is_disperse
        self._pairs_reduced = None
        self._pairs_included = None
        self._triplets_included = None
        self._triplets = None
        self._fc2 = None
        self._ifc2_ele = None
        self._ifc2_trans = None
        self._fc2_read = None
        self._fc3_read = None
        self._fc3_irred = None
        self._fc2_irred  = None
        self._shortest_vectors = None
        self._multiplicity = None
        self._rot_fc3_residue = None
        self._rot_fc3_transform = None

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

    def set_smallest_vectors(self):
        primitive = self._primitive
        supercell = self._supercell
        symprec = self._symprec
        p2s_map = self._primitive.get_primitive_to_supercell_map()
        size_super = supercell.get_number_of_atoms()
        size_prim = primitive.get_number_of_atoms()
        shortest_vectors = np.zeros((size_super, size_prim, 27, 3), dtype='double')
        multiplicity = np.zeros((size_super, size_prim), dtype='intc')

        for i in range(size_super): # run in supercell
            for j, s_j in enumerate(p2s_map): # run in unitcell
                vectors = get_equivalent_smallest_vectors(i,
                                                          s_j,
                                                          supercell,
                                                          primitive.get_cell(),
                                                          symprec)
                multiplicity[i][j] = len(vectors)
                for k, elem in enumerate(vectors):
                    shortest_vectors[i][j][k] = elem
        self._shortest_vectors = shortest_vectors
        self._multiplicity = multiplicity

    def show_drift_fc3(self, fc3=None, name="fc3"):
        if fc3 is None:
            fc3 = self._fc3_read
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
    #
    # def show_rotational_invariance_fc3(self, fc3 = None, name='fc3'):
    #     supercell = self._supercell
    #     if fc3 is None:
    #         fc3 = self._fc3_read
    #     unit_atoms = self._primitive.get_primitive_to_supercell_map()
    #     natom = supercell.get_number_of_atoms()
    #     lattice = self._primitive.get_cell()
    #     eijk = np.zeros((3,3,3), dtype="intc")
    #     eijk[0,1,2] = eijk[1,2,0] = eijk[2,0,1] = 1
    #     eijk[0,2,1] = eijk[2,1,0] = eijk[1,0,2] = -1 # epsilon matrix, which is an antisymmetric 3 * 3 * 3 tensor
    #
    #     forces = []
    #     for i, atom1 in enumerate(unit_atoms):
    #         force_on_atom = np.zeros((3,3,3), dtype=np.float)
    #         for atom2 in range(natom):
    #         #     if self._multiplicity is not None:
    #         #         vectors2 = [self._shortest_vectors[atom2, i, mm] for mm in range(self._multiplicity[atom2, i])]
    #         #     else:
    #             vectors2 = get_equivalent_smallest_vectors(atom2,
    #                                                       atom1,
    #                                                       supercell,
    #                                                       lattice,
    #                                                       self._symprec)
    #             r_frac2 = np.array(vectors2).sum(axis=0) / len(vectors2)
    #             r2 = np.dot(r_frac2, lattice)
    #             t2 = np.dot(eijk, r2)
    #             for atom3 in range(natom):
    #                 fc_temp = fc3[i, atom2, atom3]
    #                 # if self._multiplicity is not None:
    #                 #     vectors3 = [self._shortest_vectors[atom3, i, mm]
    #                 #                 for mm in range(self._multiplicity[atom3, i])]
    #                 # else:
    #                 vectors3 = get_equivalent_smallest_vectors(atom3,
    #                                                           atom1,
    #                                                           supercell,
    #                                                           lattice,
    #                                                           self._symprec)
    #                 r_frac3 = np.array(vectors3).sum(axis=0) / len(vectors3)
    #                 r3 = np.dot(r_frac3, lattice)
    #                 t3 = np.dot(eijk, r3)
    #                 force_on_atom += np.einsum("abc, ib, jc->aij", fc_temp, t2, t3)
    #         forces.append(force_on_atom)
    #     forces = np.abs(np.array(forces))
    #     max_drift = np.max(forces, axis=(0, 2, 3))
    #     print ("max rotational drift of %s:" % name), max_drift


    def show_rotational_invariance_fc3(self, fc3 = None, fc2 = None):
        if fc3 is None:
            fc3 = self._fc3_read
        if fc2 is None:
            fc2 = self._fc2_read
        supercell = self._supercell
        unit_atoms = np.unique(supercell.get_supercell_to_unitcell_map())
        natom = supercell.get_number_of_atoms()
        lattice = supercell.get_cell()
        eijk = np.zeros((3,3,3), dtype="intc")
        eijk[0,1,2] = eijk[1,2,0] = eijk[2,0,1] = 1
        eijk[0,2,1] = eijk[2,1,0] = eijk[1,0,2] = -1 # epsilon matrix, which is an antisymmetric 3 * 3 * 3 tensor
        torques = []
        for i, atom1 in enumerate(unit_atoms):
            for atom2 in range(natom):
                torque = np.zeros((3,3,3), dtype=np.float)
                t2_1 = np.einsum("cb, cav -> abv", fc2[atom1, atom2], eijk)
                t2_2 = np.einsum("ac, cbv -> abv", fc2[atom1, atom2], eijk)
                for atom3 in range(natom):
                    fc_temp = fc3[atom1, atom2, atom3]
                    vectors = get_equivalent_smallest_vectors(atom3,
                                                              atom1,
                                                              supercell,
                                                              lattice,
                                                              self._symprec)
                    r_frac = np.array(vectors).sum(axis=0) / len(vectors)
                    r = np.dot(r_frac, lattice)
                    t3 = np.einsum("abc, d, cdv -> abv", fc_temp, r, eijk)
                    torque += t3
                torque += t2_1 + t2_2
                torques.append(torque)
        torques = np.array(torques)
        print "max rotational drift of fc3:", np.abs(torques).max()

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

    def set_second_independents(self):
        symmetry = self._symmetry
        pairs = []
        for atom1 in np.unique(symmetry.mapping):
            #atom1 ranges in the first irreducible atoms
            for atom2 in np.arange(self._num_atom):
                atom2_, map_ope = symmetry.get_atom_mapping_under_sitesymmetry(atom1, atom2)
                if atom2_ < atom2:
                    continue #then this pair already exists.
                if symmetry.mapping[atom2] != atom1:
                    continue #this pair either exists or will be indexed later
                atom2_ = symmetry.get_atom_sent_by_operation(atom1, symmetry.mapping_operations[atom2])
                atom2_, _ = symmetry.get_atom_mapping_under_sitesymmetry(atom1, atom2_)
                if atom2_ < atom2:
                    continue
                pairs.append((atom1, atom2))
        self._pairs = pairs

    def set_pair_included(self, pair_inclusion=None):
        if pair_inclusion is None:
            pair_inclusion = np.ones(len(self._pairs), dtype=bool)
        self._pairs_included = pair_inclusion

    def set_triplet_included(self, triplet_inclusion=None):
        if triplet_inclusion is None:
            triplet_inclusion = np.ones(len(self._triplets), dtype=bool)
        self._triplets_included = triplet_inclusion

    def get_fc2_coefficients(self):
        self._ifc2_map, self._coeff2 = get_fc2_coefficients(self._pairs, self._symmetry)

    def fc2_coefficient(self, atom1=None, atom2=None):
        if atom1 is not None:
            if atom2 is not None:
                return self._symmetry.tensor2[self._coeff2[atom1, atom2]].swapaxes(-1, -2)
            else:
                return self._symmetry.tensor2[self._coeff2[atom1, :]].swapaxes(-1,-2)
        else:
            if atom2 is not None:
                return self._symmetry.tensor2[self._coeff2[:, atom2]].swapaxes(-1,-2)
            else:
                return self._symmetry.tensor2[self._coeff2].swapaxes(-1, -2)

    def get_irreducible_fc2_components_with_spg(self):
        if self._symmetry.tensor2 is None:
            self._symmetry.set_tensor2()
        ifc2_ele, ifc2_trans = \
            get_fc2_spg_invariance(self._pairs,
                                   self._symmetry)
        num_irred = [len(ele) for ele in ifc2_ele]

        ifc2_ele_array = np.zeros(sum(num_irred), dtype=int)
        trans = np.zeros((len(ifc2_ele), 9, sum(num_irred)), dtype="float")
        for i, length in enumerate(num_irred):
            start = sum(num_irred[:i])
            end = start + length
            ifc2_ele_array[start:end] = np.array(ifc2_ele[i]) + i * 9
            trans[i, :, start:end] = ifc2_trans[:, start:end]
        self._ifc2_ele = ifc2_ele_array
        self._ifc2_trans = trans

    def get_fc2_translational_invariance(self):
        if self._fc2_read is not None:
            fc2_reduced_pair = np.array([self._fc2_read[pair] for pair in self._pairs_reduced])
            fc2_irr_orig = fc2_reduced_pair.flatten()[self._ifc2_ele]
        irreducible_tmp, transform_tmp = \
            get_fc2_translational_invariance(self._supercell,
                                             self._ifc2_trans,
                                             self._coeff2,
                                             self._ifc2_map,
                                             precision=self._precision)
        self._ifc2_ele = self._ifc2_ele[irreducible_tmp]
        self._ifc2_trans = mat_dot_product(self._ifc2_trans, transform_tmp, is_sparse=True)
        #checking the results of gaussian elimination
        if self._fc2_read is not None:
            fc2_irr_new = fc2_irr_orig[irreducible_tmp]
            check_descrepancy(np.dot(transform_tmp, fc2_irr_new), fc2_irr_orig, info='translational')


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
        self._ifc2_trans = mat_dot_product(self._ifc2_trans, transform_tmp, is_sparse=True)
        if self._fc2_read is not None:
            fc2_irr_new = fc2_irr_orig[irreducible_tmp]
            check_descrepancy(np.dot(transform_tmp, fc2_irr_new), fc2_irr_orig, info='rotational')

    def set_trim_fc2(self):
        if self._fc2_read is not None:
            fc2_reduced_pair = np.array([self._fc2_read[pair] for pair in self._pairs_reduced])
            fc2_irr_orig = fc2_reduced_pair.flatten()[self._ifc2_ele]

        irreducible_tmp, transform_tmp = \
            get_trim_fc2(self._supercell,
                         self._ifc2_trans,
                         self._coeff2,
                         self._ifc2_map,
                         symprec=self._symprec,
                         precision=self._precision,
                         pairs_included=self._pairs_included)
        self._ifc2_ele = self._ifc2_ele[irreducible_tmp]
        self._ifc2_trans = mat_dot_product(self._ifc2_trans, transform_tmp, is_sparse=True)
        if self._fc2_read is not None:
            fc2_irr_new = fc2_irr_orig[irreducible_tmp]
            check_descrepancy(np.dot(transform_tmp, fc2_irr_new), fc2_irr_orig, info='trimming')

    @profile
    def set_fc2_irreducible_elements(self, is_trans_inv=False, is_rot_inv=False, is_md=False):
        if self._symmetry.tensor2 is None:
            self._symmetry.set_tensor2(True)
        equivalent_atoms = self._symmetry.mapping
        print "Under the system symmetry"
        print "Number of first independent atoms: %4d" % len(np.unique(equivalent_atoms))
        self.set_second_independents()
        print "Number of irreducible pairs: %4d" %len(self._pairs)

        if (self._cutoff is not None) and (self._cutoff.get_cutoff_radius()is not None):
            pair_inclusion = self._cutoff.get_pair_inclusion(pairs=self._pairs)
            self.set_pair_included(pair_inclusion)
            print "The artificial cutoff reduces number of irreducible pairs from %4d to %4d"\
                  %(len(self._pairs), np.sum(self._pairs_included))
        else:
            self.set_pair_included()
        print "Number of independent fc2 components: %d" % (len(self._pairs) * 9)

        self.get_irreducible_fc2_components_with_spg() # space group operation redundance
        print "Point group invariance reduces independent fc2 components to %d" % (self._ifc2_trans.shape[-1])
        print "Calculating transformation coefficients..."
        self.get_fc2_coefficients()


        sys.stdout.flush()
        if not is_md:
            if is_trans_inv:
                self.get_fc2_translational_invariance()
                print "Translational invariance further reduces independent fc2 components to %d" %len(self._ifc2_ele)
            if is_rot_inv:
                self.get_fc2_rotational_invariance()
                print "Rotational invariance further reduces independent fc2 components to %d" %len(self._ifc2_ele)
            if not np.all(self._pairs_included):
                self.set_trim_fc2()
                print "Interaction distance cutoff further reduces independent fc2 components to %d" %len(self._ifc2_ele)
            print "Independent fc2 components calculation completed"
            if DEBUG:
                from mdfc.file_IO import read_fc2_from_hdf5
                fc2 = read_fc2_from_hdf5("fc2.hdf5")
                fc2_reduced = np.array([fc2[pai] for pai in self._pairs_reduced])
                fc2p = fc2_reduced.flatten()[self._ifc2_ele]
                pp = np.einsum('ijkl, ijl-> ijk', self._coeff2, fc2_reduced[self._ifc2_map].reshape(self._num_atom, self._num_atom, 9)).reshape(self._num_atom, self._num_atom, 3, 3)
            sys.stdout.flush()

    def tune_fc2(self, is_minimize_relative_error=False, log_level=1):
        self._fc2 = np.zeros_like(self._fc2_read)
        len_element = len(self._ifc2_ele)
        first_atoms = np.unique(self._supercell.get_supercell_to_unitcell_map())
        transform = np.zeros((len(first_atoms), self._num_atom, 9, len_element), dtype='double')
        for first, i in enumerate(first_atoms):
            for j in range(self._num_atom):
                transform[first,j] = np.dot(self._coeff2[i,j], self._ifc2_trans[self._ifc2_map[i, j]])
        transform2 = transform.reshape(-1, len_element)
        fc2_read_flatten = self._fc2_read[first_atoms].flatten()
        if is_minimize_relative_error:
            fc_scale = np.zeros_like(self._fc2_read[first_atoms])
            transform2 = np.zeros_like(transform)
            for i, j in np.ndindex((len(first_atoms), self._num_atom)):
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
            self._fc2_irred = lsqr_results[0]
            error = lsqr_results[3] / len(first_atoms)
            fc_tuned = mat_dot_product(transform, self._fc2_irred, is_sparse=True)
        except ImportError:
            transform_pinv = np.linalg.pinv(transform2)
            self._fc2_irred = np.dot(transform_pinv, fc2_read_flatten)
            fc_tuned = np.dot(transform, self._fc2_irred)
            error = np.sqrt(np.sum((fc2_read_flatten - fc_tuned.flatten())**2)) / len(first_atoms)
        self.distribute_fc2()
        print "FC2 tunning process using the least-square method has completed"
        print "    with least square error: %f (eV/A^2)" %error
        if log_level == 2:
            print "The comparison between original and the tuned force constants is plot and saved to f2-tune_compare.pdf"
            plt.figure()
            plt.scatter(fc2_read_flatten, fc_tuned, color='red', s=3)
            plt.plot(np.array([fc2_read_flatten.min(), fc2_read_flatten.max()]),
                     np.array([fc2_read_flatten.min(), fc2_read_flatten.max()]), color='blue')
            threshold = 10 ** np.rint(np.log10(np.abs(fc2_read_flatten).max() / 1e3))
            plt.yscale('symlog', linthreshy=threshold)
            plt.xscale('symlog', linthreshx=threshold)
            plt.xlabel("Original fc2 (eV/A^2)")
            plt.ylabel("Tuned fc2 (eV/A^2)")
            plt.savefig("fc_tune_compare.pdf")
        write_fc2_hdf5(self._fc2, filename='fc2-tuned.hdf5')


    def tune_fc3(self):
        len_element = len(self._ifc3_ele)
        first_atoms = np.unique(self._supercell.get_supercell_to_unitcell_map())
        natom = self._num_atom
        fc3_read = self._fc3_read[first_atoms].flatten()
        print "Solving the least square problem..."
        if self._is_disperse:
            transform_sparse = dok_matrix((len(first_atoms)*self._num_atom*self._num_atom*27, len_element),dtype='double')
            for first in range(len(first_atoms)):
                for j, k in np.ndindex((self._num_atom, self._num_atom)):
                    first_index_tmp = first * natom * natom * 27 + j * natom * 27 + k * 27
                    transform_tmp = mat_dot_product(self._coeff3[first,j, k],
                                                    self._ifc3_trans[self._ifc3_map[first, j, k]],
                                                    is_sparse=True)
                    non_zero = np.where(np.abs(transform_tmp) > self._precision / 1e2)
                    for (m, n) in zip(*non_zero):
                        transform_sparse[first_index_tmp + m, n] = transform_tmp[m, n]
            transform_sparse = transform_sparse.tocoo()
            transform = transform_sparse.transpose().dot(transform_sparse).toarray()
            fc3_read_transformed = transform_sparse.transpose().dot(fc3_read)
        else:
            transform = np.zeros((len(first_atoms), self._num_atom,self._num_atom, 27, len_element), dtype='double')
            for first in range(len(first_atoms)):
                for j, k in np.ndindex((self._num_atom, self._num_atom)):
                    transform[first, j, k] = mat_dot_product(self._coeff3[first,j, k],
                                                             self._ifc3_trans[self._ifc3_map[first, j, k]],
                                                             is_sparse=True)
            transform2 = transform.reshape(-1, len_element)
            non_zero = np.where(np.abs(transform2) > self._precision / 1e2)
            transform_sparse = coo_matrix((transform2[non_zero], non_zero), shape=transform2.shape)
            transform = transform_sparse.transpose().dot(transform_sparse).toarray()
            fc3_read_transformed = transform_sparse.transpose().dot(fc3_read)

        if self._rot_fc3_transform is not None:
            len_constraint = self._rot_fc3_transform.shape[0]
            transform_extended = np.zeros((len_element + len_constraint,
                                            len_element + len_constraint),dtype='double')
            transform_extended[:len_element, :len_element] = transform
            transform_extended[len_element:, :len_element] = self._rot_fc3_transform[:]
            transform_extended[:len_element, len_element:] = self._rot_fc3_transform[:].T
            transform = transform_extended
            fc3_read_transformed = np.hstack((fc3_read_transformed, self._rot_fc3_residue))

        self._fc3_irred = np.dot(np.linalg.pinv(transform), fc3_read_transformed)[:len_element]
        fc_tuned = mat_dot_product(transform_sparse, self._fc3_irred, is_sparse=True)

        error = np.sqrt(np.sum((fc_tuned - fc3_read)**2))\
                / len(first_atoms) / self._num_atom
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
        if self._symmetry.tensor3 is None:
            self._symmetry.set_tensor3()
        self.set_third_independents()
        print "Number of irreducible triplets found: %d" %(len(self._triplets))
        print "Number of 3rd IFC: %d" %(27 * len(self._triplets))
        if (self._cutoff is not None) and (self._cutoff.get_cutoff_radius() is not None):
            triplets_inclusion = self._cutoff.get_triplet_inclusion(triplets=self._triplets)
            self.set_triplet_included(triplets_inclusion)
            print "The artificial cutoff reduces number of irreducible pairs from %4d to %4d"\
                  %(len(self._triplets), np.sum(self._triplets_included))
        else:
            self.set_triplet_included()
        self.get_irreducible_fc3_components_with_spg()
        print "spg invariance reduces 3rd IFC to %d" % (len(self._ifc3_ele))
        print "Calculating fc3 coefficient..."
        self.get_fc3_coefficients()

        if not np.all(self._triplets_included):
            self.set_trim_fc3()
            print "Interaction distance cutoff reduces independent fc3 components to %d" %len(self._ifc3_ele)
        if is_trans_inv:
            print "Reducing the number of fc3s by considering translational invariance"
            self.get_fc3_translational_invariance()
            print "translational invariance reduces 3rd IFC to %d"%(len(self._ifc3_ele))
        if is_rot_inv and self._fc2 is not None:
            print "Getting the number of constraints on fc3 by considering rotational invariance"
            self.get_fc3_rotational_invariance(self._fc2)
        self._num_irred_fc3 = len(self._ifc3_ele)
        sys.stdout.flush()

    @timeit
    def set_third_independents(self):
        symmetry = self._symmetry
        self._triplets = []
        for atom1 in np.unique(symmetry.mapping):
            #atom1 ranges in the first irreducible atoms
            for atom2 in np.arange(self._num_atom):
                atom2_, _ = symmetry.get_atom_mapping_under_sitesymmetry(atom1, atom2)
                if atom2_ != atom2:
                    continue #then this pair already exists.
                bond_symmetry = symmetry.get_site_symmetry_at_atoms([atom1, atom2])
                for atom3 in np.arange(self._num_atom):
                    atom3_, _ = symmetry.get_atom_mapping_under_sitesymmetry(atom1, atom3, bond_symmetry)
                    if atom3_ != atom3:
                        continue
                    #Use permutations to check if the triplet has equivalence in previous ones
                    is_exist = False
                    for _atom1, _atom2, _atom3 in list(permutations((atom1, atom2, atom3)))[1:]:
                        atom1_ = symmetry.mapping[_atom1]
                        nope = symmetry.mapping_operations[_atom1]
                        if atom1_ < atom1:
                            is_exist = True
                            break
                        elif atom1_ == atom1:
                            atom2_ = symmetry.get_atom_sent_by_operation(_atom2, nope)
                            atom2_, mapope = symmetry.get_atom_mapping_under_sitesymmetry(atom1, atom2_)
                            if atom2_ < atom2:
                                is_exist = True
                                break
                            elif atom2_ == atom2:
                                atom3_ = symmetry.get_atom_sent_by_operation(_atom3, nope)
                                atom3_ = symmetry.get_atom_sent_by_site_sym(atom1, atom3_, mapope)
                                atom3_, _ = symmetry.get_atom_mapping_under_sitesymmetry(atom1, atom3_, bond_symmetry)
                                if atom3_ < atom3:
                                    is_exist = True
                                    break
                    if not is_exist:
                        self._triplets.append((atom1, atom2, atom3))


    @timeit
    def get_fc3_coefficients(self):
        self._coeff3, self._ifc3_map =\
            get_fc3_coefficients(self._triplets,
                                 self._symmetry)

    def fc3_coefficient(self, atom1, atom2=None, atom3=None):
        if atom2 is None and atom3 is None:
            coeff = self._coeff3[atom1]
        elif atom2 is not None and atom3 is None:
            coeff = self._coeff3[atom1, atom2, :]
        elif atom2 is None and atom3 is not None:
            coeff = self._coeff3[atom1, :, atom2]
        else: #atom2 is not None and atom3 is not None
            coeff = self._coeff3[atom1, atom2, atom3]
        tensor = self._symmetry.tensor3[coeff]
        return tensor.swapaxes(-1, -2)

    @timeit
    def distribute_fc3(self):
        coeff = self._coeff3
        ifcmap = self._ifc3_map
        trans = self._ifc3_trans
        assert self._fc3_irred is not None
        s2u_map = self._supercell.get_supercell_to_unitcell_map()
        positions = self._supercell.get_scaled_positions()
        # distribute all the fc3s
        print "Distributing fc3..."
        fc3 = np.zeros((self._num_atom, self._num_atom, self._num_atom, 3,3,3), dtype="double")
        for i, atom1 in enumerate(np.unique(s2u_map)):
            for atom2 in np.arange(self._num_atom):
                for atom3 in np.arange(self._num_atom):
                    num_triplet = ifcmap[i, atom2, atom3]
                    trans_temp = mat_dot_product(coeff[i, atom2, atom3], trans[num_triplet], is_sparse=True)
                    fc3[atom1, atom2, atom3] = np.dot(trans_temp, self._fc3_irred).reshape(3,3,3)
        for atom1 in range(self._num_atom):
            if atom1 in s2u_map:
                continue
            atom1_ = s2u_map[atom1]
            disp = positions[atom1] - positions[atom1_]
            for atom2 in range(self._num_atom):
                pos_atom2 = positions[atom2] - disp
                atom2_ = get_index_of_atom(pos_atom2, positions, self._symprec)
                for atom3 in range(self._num_atom):
                    pos_atom3 = positions[atom3] - disp
                    atom3_ = get_index_of_atom(pos_atom3, positions, self._symprec)
                    fc3[atom1, atom2, atom3] = fc3[atom1_, atom2_, atom3_]
        self._fc3 = fc3

    @timeit
    def distribute_fc2(self):
        coeff = self._coeff2
        ifcmap = self._ifc2_map
        trans = self._ifc2_trans
        assert self._fc2_irred is not None
        s2u_map = self._supercell.get_supercell_to_unitcell_map()
        scaled_positions = self._supercell.get_scaled_positions()
        # distribute all the fc2s
        print "Distributing fc2..."
        fc2 = np.zeros((self._num_atom, self._num_atom, 3, 3), dtype="double")
        for atom1 in np.unique(s2u_map):
            for atom2 in np.arange(self._num_atom):
                num_triplet = ifcmap[atom1, atom2]
                trans_temp = mat_dot_product(coeff[atom1, atom2], trans[num_triplet], is_sparse=True)
                fc2[atom1, atom2] = np.dot(trans_temp, self._fc2_irred).reshape(3, 3)
        for atom1 in range(self._num_atom):
            if atom1 in s2u_map:
                continue
            ip = s2u_map[atom1]
            l = scaled_positions[atom1] - scaled_positions[ip]
            for atom2 in range(self._num_atom):
                disp2 = scaled_positions[atom2] - l - scaled_positions
                jp = np.where(np.all(np.abs(disp2 - np.rint(disp2)) < self._symprec, axis=-1))[0][0]
                fc2[atom1, atom2] = fc2[ip, jp]
        self._fc2 = fc2

    @timeit
    def get_irreducible_fc3_components_with_spg(self):
        ifc3_ele, ifc3_trans = \
            get_fc3_spg_invariance(self._triplets, self._symmetry)
        num_irred = [len(ind) for ind in ifc3_ele]
        trans = np.zeros((len(self._triplets), 27, sum(num_irred)), dtype="float")
        for i in range(len(self._triplets)):
            start = sum(num_irred[:i])
            length = num_irred[i]
            trans[i, :, start:start + length] = ifc3_trans[:, start:start + length]
        self._ifc3_trans = trans
        self._ifc3_ele = sum([map(lambda x: x + 27 * i, ele) for (i, ele) in enumerate(ifc3_ele)], [])
        if DEBUG:
            fc3_read = self._fc3_read
            fc3_reduced_triplets = np.double([fc3_read[index] for index in self._triplets])
            fc3_reduced = fc3_reduced_triplets.flatten()[self._ifc3_ele]
            fc3_reduced_triplets2 = np.zeros_like(fc3_reduced_triplets)
            for i in range(len(fc3_reduced_triplets)):
                fc3_reduced_triplets2[i] = mat_dot_product(self._ifc3_trans[i], fc3_reduced, is_sparse=self._is_disperse)
            diff = fc3_reduced_triplets2 - fc3_reduced_triplets.reshape(-1, 27)
            print np.abs(diff).max()

    def get_fc3_translational_invariance(self):
        if self._fc3_read is not None:
            fc3_reduced_pair = np.array([self._fc3_read[triplet] for triplet in self._triplets])
            fc3_irr_orig = fc3_reduced_pair.flatten()[self._ifc3_ele]
        irreducible_tmp, transform_tmp = \
            get_fc3_translational_invariance(self._supercell,
                                             self._ifc3_trans,
                                             self._coeff3,
                                             self._ifc3_map,
                                             self._precision)
        self._ifc3_ele = self._ifc3_ele[irreducible_tmp]
        if self._is_disperse:
            for i in range(len(self._ifc3_trans)):
                transform2 = mat_dot_product(self._ifc3_trans[i], transform_tmp, is_sparse=True)
                self._ifc3_trans[i] = mat_dense_to_sparse(transform2)
        else:
            self._ifc3_trans = mat_dot_product(self._ifc3_trans, transform_tmp, is_sparse=True)
        if self._fc3_read is not None:
            fc3_irr_new = fc3_irr_orig[irreducible_tmp]
            check_descrepancy(np.dot(transform_tmp, fc3_irr_new), fc3_irr_orig, info="translational")

    def get_fc3_rotational_invariance(self, fc2):
        if self._fc3_read is not None:
            fc3_reduced_pair = np.array([self._fc3_read[triplet] for triplet in self._triplets])
            fc3_irr_orig = fc3_reduced_pair.flatten()[self._ifc3_ele]
        rot_fc3_transform, rot_fc3_residue =\
            get_fc3_rotational_invariance(fc2,
                                          self._supercell,
                                          self._ifc3_trans,
                                          self._coeff3,
                                          self._ifc3_map,
                                          self._symprec,
                                          precision=self._precision)

        self._rot_fc3_transform = rot_fc3_transform
        self._rot_fc3_residue = rot_fc3_residue
        if self._fc3_read is not None:
            check_descrepancy(np.dot(rot_fc3_transform, fc3_irr_orig), rot_fc3_residue, info="rotational fc3")

    def set_trim_fc3(self):
        if self._fc3_read is not None:
            fc3_reduced_pair = np.array([self._fc3_read[triplet] for triplet in self._triplets])
            fc3_irr_orig = fc3_reduced_pair.flatten()[self._ifc3_ele]
        irreducible_tmp, transform_tmp = \
            get_trim_fc3(self._supercell,
                         self._ifc3_trans,
                         self._triplets,
                         symprec=self._symprec,
                         precision=self._precision,
                         triplets_included=self._triplets_included)
        self._ifc3_ele = self._ifc3_ele[irreducible_tmp]
        if self. _is_disperse:
            for i in range(len(self._ifc3_trans)):
                transform2 = mat_dot_product(self._ifc3_trans[i], transform_tmp, is_sparse=True)
                self._ifc3_trans[i] = mat_dense_to_sparse(transform2)
        else:
            self._ifc3_trans = mat_dot_product(self._ifc3_trans, transform_tmp, is_sparse=True)
        if self._fc3_read is not None:
            fc3_irr_new = fc3_irr_orig[irreducible_tmp]
            check_descrepancy(np.dot(transform_tmp, fc3_irr_new), fc3_irr_orig, info="trimming")

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
    # from phonopy.structure.symmetry import Symmetry
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