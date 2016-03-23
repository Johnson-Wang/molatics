import numpy as np
from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors
from fc2 import get_atom_mapping, get_pairs_with_permute, get_fc2_coefficients, get_fc2_spg_invariance,\
    get_fc2_translational_invariance, get_fc2_rotational_invariance
from fc3 import get_bond_symmetry, get_irreducible_triplets_with_permute, get_fc3_coefficients, get_fc3_spg_invariance,\
    get_fc3_translational_invariance, get_fc3_rotational_invariance

from realmd.information import timeit

class ForceConstants():
    def __init__(self, atoms, symmetry):
        self._symmetry = symmetry
        self._atoms = atoms
        self._ind1 = None
        self._ind2 = None
        self._ind3 = None
        self._atom_number = len(atoms.get_scaled_positions())
        self._positions = atoms.get_scaled_positions()
        self._symprec = symmetry.get_symmetry_tolerance()
        self._lattice = atoms.get_cell().T
        self._pairs_reduced = None
        self._is_pairs_included = None

    def set_first_independents(self):
        sym = self._symmetry
        independents = {}
        independents['atoms'] = sym.get_independent_atoms() # independent atoms
        independents['natoms'] = len(independents['atoms']) # number of independent atoms
        sym_operations = sym.get_symmetry_operations()
        independents['rotations'] = sym_operations['rotations']
        independents['translations'] = sym_operations['translations']
        independents['noperations'] = len(sym_operations['rotations'])
        # rotations and translations forms all the operational
        independents['mappings'] = sym.get_map_atoms()
        independents['mapping_operations'] = sym.get_map_operations()
        self._ind1 = independents

    def set_second_independents(self, pair_included=None):
        "pair_included is a natom_super * natom_super matrix determing whether fc2 between any two atoms in the supercell is included"
        symprec = self._symmetry.get_symmetry_tolerance()
        positions = self._atoms.get_scaled_positions()
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
        mappings = np.zeros((self._ind1['natoms'], self._atom_number), dtype="intc")
        mapping_operations = np.zeros((self._ind1['natoms'], self._atom_number), dtype="intc")
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
        is_pairs_included = np.ones(len(self._pairs_reduced), dtype=bool)
        if pair_inclusion is not  None:
            for i, (atom1, atom2) in enumerate(self._pairs_reduced):
                is_pairs_included[i] = pair_inclusion[atom1, atom2]
        self._is_pairs_included = is_pairs_included
        self._pair_included = np.array(self._pairs_reduced)[np.where(self._is_pairs_included)]

    def set_triplet_reduced_included(self, triplet_inclusion=None):
        is_triplet_included = np.ones(len(self._triplets_reduced), dtype=bool)
        if triplet_inclusion is not None:

            for i, (atom1, atom2, atom3) in enumerate(self._triplets_reduced):
                is_triplet_included[i] = triplet_inclusion[i]
        self._is_triplets_included = is_triplet_included
        self._triplet_included = np.array(self._triplets_reduced)[np.where(is_triplet_included)]

    def get_irreducible_fc2s_with_permute(self):
        ind1 = self._ind1
        ind2 = self._ind2
        self._pair_mappings, self._pair_transforms = \
            get_pairs_with_permute(self._pairs,
                                   self._lattice.copy().astype("double"),
                                   self._atoms.get_scaled_positions(),
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
                                 self._lattice.copy().astype("double"),
                                 self._atoms.get_scaled_positions(),
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
            get_fc2_spg_invariance(np.array(self._pairs_reduced),
                                   self._is_pairs_included,
                                   self._atoms.get_scaled_positions(),
                                   ind1['rotations'],
                                   ind1['translations'],
                                   ind1['mappings'],
                                   ind2['rotations'],
                                   ind2['noperations'],
                                   ind2['mappings'],
                                   self._lattice,
                                   self._symmetry.get_symmetry_tolerance())

    def get_fc2_translational_invariance(self):
        self._ifc2_ele, self._ifc2_trans = \
            get_fc2_translational_invariance(self._ifc2_trans,
                                         self._ifc2_ele,
                                         self._coeff2,
                                         self._ifc2_map,
                                         self._ind1['atoms'],
                                         self._atoms.get_scaled_positions())

    def get_fc2_rotational_invariance(self, unitcell):
        self._ifc2_ele, self._ifc2_trans = \
            get_fc2_rotational_invariance(self._atoms,
                                          unitcell,
                                          self._ifc2_trans,
                                          self._ifc2_ele,
                                          self._coeff2,
                                          self._ifc2_map,
                                          self._ind1['atoms'],
                                          self._symmetry.get_symmetry_tolerance())

    @timeit
    def set_third_independents(self):
        symprec = self._symmetry.get_symmetry_tolerance()
        positions = self._atoms.get_scaled_positions()
        ind1 = self._ind1
        ind2 = self._ind2
        if ind2 == None:
            self.set_second_independents()
        ind3 = {}
        nind_atoms = np.zeros_like(ind2['atoms'])
        ind_mappings = np.zeros(ind2['atoms'].shape  + (self._atom_number,), dtype="intc")
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
            self._coeff3 = np.zeros((self._atom_number, self._atom_number, self._atom_number, 27, 27), dtype="double")
            self._ifc3_map = np.zeros((self._atom_number, self._atom_number, self._atom_number), dtype="intc")
            _mdfc.get_fc3_coefficients(self._coeff3,
                                         self._ifc3_map,
                                         np.array(self._triplets).astype("intc"),
                                         self._triplet_mappings.astype("intc"),
                                         self._triplet_transforms.astype("double"),
                                         self._lattice.copy().astype("double"),
                                         self._positions.astype("double"),
                                         ind1['rotations'].copy().astype("intc"),
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
    def get_irreducible_fc3_components_with_spg(self, lang="py"):
        ind1 = self._ind1
        ind2 = self._ind2
        ind3 = self._ind3
        if lang == "py":
            self._ifc3_ele, self._ifc3_trans = \
                get_fc3_spg_invariance(self._triplets_reduced,
                                       self._is_triplets_included,
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
                                         self._is_triplets_included.copy().astype("int8"),
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

    def get_fc3_translational_invariance(self):
        self._ifc3_ele, self._ifc3_trans = \
            get_fc3_translational_invariance(self._ifc3_trans,
                                             self._ifc3_ele,
                                             self._coeff3,
                                             self._ifc3_map,
                                             self._ind1['atoms'],
                                             self._ind2['atoms'],
                                             self._ind2['natoms'],
                                             self._positions)

    def get_fc3_rotational_invariance(self, fc2):
        self._ifc3_ele, self._ifc3_trans = \
            get_fc3_rotational_invariance(fc2,
                                          self._ifc3_trans,
                                         self._ifc3_ele,
                                         self._coeff3,
                                         self._ifc3_map,
                                         self._ind1['atoms'],
                                         self._ind2['atoms'],
                                         self._ind2['natoms'],
                                         self._atoms,
                                         self._symprec)




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


def similarity_transformation(rot, mat):
    """ R x M x R^-1 """
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))

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