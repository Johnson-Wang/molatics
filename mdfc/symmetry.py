import numpy as np
import spglib
from phonopy.structure.cells import Primitive
from itertools import permutations
from mdfc.fcmath import similarity_transformation
class Symmetry():
    def __init__(self, cell, symprec=1e-6):
        self.cell = cell
        self.positions = self.cell.get_scaled_positions()
        self.symprec = symprec
        self.rot_inv = None
        self.rot_mult = None
        self.tensor2 = None
        self.tensor3 = None
        self.site_symmetries = {}
        self.symmetry_operations = \
            spglib.get_symmetry(cell=cell, symprec=symprec)
        self.get_pointgroup_operations()
        # self.get_unique_translations()
        self.rotations = np.zeros(len(self.symmetry_operations['rotations']), dtype="intc")
        self.translations = self.symmetry_operations['translations']
        for i, rot in enumerate(self.symmetry_operations['rotations']):
            self.rotations[i] = np.all(self.pointgroup_operations == rot, axis=(1,2)).argmax()

        self.mapping = self.symmetry_operations['equivalent_atoms']
        self.mapping_operations = np.zeros(len(self.positions), dtype='intc')
        for i, pos in enumerate(self.positions):
            pos2 = np.dot(self.symmetry_operations['rotations'], pos) + self.symmetry_operations['translations']
            diff = self.positions[self.mapping[i]] - pos2
            diff -= np.rint(diff)
            self.mapping_operations[i] = np.where(np.all(np.abs(diff) < self.symprec, axis=1))[0][0]

    def get_pointgroup_operations(self):
        rotations = [tuple(rot.flatten()) for rot in self.symmetry_operations['rotations']]
        pointgroup_operations = np.intc(np.unique(rotations).reshape(-1, 3, 3))
        for i, rot in enumerate(pointgroup_operations):
            if (rot == np.eye(3)).all():
                pointgroup_operations[i] = pointgroup_operations[0]
                pointgroup_operations[0] = np.eye(3)
        # to ensure the first element is the I matrix
        self.pointgroup_operations = pointgroup_operations

    def get_atom_mapping_under_sitesymmetry(self, center_atom, atom, symmetries=None):
        """atom is mapped to a lower(or equal) index under the sitesymmetry at center_atom"""
        if symmetries is None:
            symmetries = self.get_site_symmetry(center_atom)
        positions = self.positions - self.positions[center_atom]
        if len(symmetries) == 1: # speeding up the case of no symmetries
            return atom, 0
        p = positions[atom]
        pos2 = self.pointgroup_operations[symmetries].dot(p) # shape: [nsym, 3]
        diff = pos2[np.newaxis] - positions[:atom+1, np.newaxis] # shape[iatom+1, nsym,3]
        diff -= np.rint(diff)
        iatoms, isyms = np.where(np.all(np.abs(diff) < self.symprec, axis=-1))
        return iatoms[0], symmetries[isyms[0]]

    def set_multiplication_table(self):
        num_rot = len(self.pointgroup_operations)
        self.rot_mult = np.zeros((num_rot, num_rot), dtype="intc")
        self.rot_inv = np.zeros(num_rot, dtype='intc')
        for i, pg1 in enumerate(self.pointgroup_operations):
            for j, pg2 in enumerate(self.pointgroup_operations):
                pg = np.dot(pg1, pg2)
                k = np.all(pg==self.pointgroup_operations, axis=(1,2)).argmax()
                self.rot_mult[i, j] = k
                if k == 0: # rot1.dot.rot2 == I
                    self.rot_inv[i] = j

    def rot_multiply(self, i, j):
        if self.rot_mult is None:
            self.set_multiplication_table()
        return self.rot_mult[i,j]

    def rot_inverse(self, i):
        if self.rot_inv is None:
            self.set_multiplication_table()
        return self.rot_inv[i]

    def get_site_symmetry(self, atom_number):
        if atom_number not in self.site_symmetries.keys():
            self.set_site_symmetry(atom_number)
        return self.site_symmetries[atom_number]

    def set_site_symmetry(self, atom_number):
        pos = self.cell.get_scaled_positions()[atom_number]
        symprec = self.symprec
        rot = self.symmetry_operations['rotations']
        trans = self.symmetry_operations['translations']
        site_symmetries = []

        for i, (r, t) in enumerate(zip(rot, trans)):
            rot_pos = np.dot(pos, r.T) + t
            diff = pos - rot_pos
            if (abs(diff - np.rint(diff)) < symprec).all():
                site_symmetries.append(self.rotations[i])
        self.site_symmetries[atom_number] = site_symmetries

    def get_bond_symmetry(self,
                          atom_center,
                          atom_disp):
        """
        Bond symmetry is the symmetry operations that keep the symmetry
        of the cell containing two fixed atoms.
        """
        pos = self.positions
        symmetries = self.get_site_symmetry(atom_center)
        if len(symmetries) == 1:
            return symmetries
        if atom_center == atom_disp:
            return symmetries
        bond_sym = []
        for i in symmetries:
            rot = self.pointgroup_operations[i]
            rot_pos = (np.dot(pos[atom_disp] - pos[atom_center], rot.T) +
                       pos[atom_center])
            diff = pos[atom_disp] - rot_pos
            if (abs(diff - diff.round()) < self.symprec).all():
                bond_sym.append(i)
        return np.intc(bond_sym)

    def get_site_symmetry_at_atoms(self,
                                   atoms):
        """
        Site symmetry at atoms (either 1 or multiple atoms)
        """

        if atoms == None:
            return np.arange(len(self.pointgroup_operations)).astype('intc')
        elif type(atoms) in [list, tuple, np.ndarray]:
            atom_center = atoms[0]
            symmetries = self.get_site_symmetry(atom_center)
            rotations = self.pointgroup_operations[symmetries]
            pos = self.positions - self.positions[atom_center]
            diff = np.array([np.dot(rotations, pos[atom]) - pos[atom] for atom in atoms]) # shape[natoms, nsym, 3]
            diff -= np.rint(diff)
            indices = np.where(np.all(np.abs(diff) < self.symprec, axis=(0,2))) # over natoms and direction
            return symmetries[indices]
        else: # atoms is an integer
            return self.get_site_symmetry(atoms)

    def set_tensor2(self):
        lattice = self.cell.get_lattice().T
        self.tensor2 = np.zeros((len(self.pointgroup_operations)*2, 9, 9), dtype='double')
        nopes = len(self.pointgroup_operations)
        for i, rot in enumerate(self.pointgroup_operations):
            rot_cart = similarity_transformation(lattice, rot)
            tensor2 = np.kron(rot_cart, rot_cart)
            self.tensor2[i] = tensor2
            self.tensor2[i+nopes] = tensor2.reshape(3,3,9).swapaxes(0,1).reshape(9,9)

    def set_tensor3(self):
        """Get the transformation tensor3 of 3rd anharmonic force constants
        Rot.Perm(Phi) = Psi*, where Phi* is the known one.
        """
        lattice = self.cell.get_lattice().T
        nopes = len(self.pointgroup_operations)
        self.tensor3 = np.zeros((nopes*6, 27,27), dtype='double')
        for i, rot in enumerate(self.pointgroup_operations):
            rot_cart = similarity_transformation(lattice, rot)
            tensor3 = np.kron(np.kron(rot_cart, rot_cart), rot_cart).reshape(3,3,3,27)
            for j, perm in enumerate(permutations("ijk")):
                self.tensor3[i+j*nopes] = np.einsum("ijkl->%sl"%perm, tensor3).reshape(27,27)

    def get_all_operations_at_star(self, atom):
        star = self.mapping[atom]
        map_operations = []
        for j, (r, t) in enumerate(
                zip(self.symmetry_operations['rotations'], self.symmetry_operations['translations'])):
            diff = np.dot(self.positions[atom], r.T) + t - self.positions[star]
            if (abs(diff - np.rint(diff)) < self.symprec).all():
                map_operations.append(j)
        return map_operations

    def get_atom_sent_by_operation(self, orig_atom, numope):
        r = self.symmetry_operations['rotations'][numope]
        t = self.symmetry_operations['translations'][numope]
        rot_pos = np.dot(self.positions[orig_atom], r.T) + t
        diff = self.positions - rot_pos
        atom2 = np.where(np.all(np.abs(diff - np.rint(diff)) < self.symprec, axis=1))[0][0]
        return atom2

    def get_atom_sent_by_site_sym(self, atom, center, nrot):
        if nrot == 0:
            return atom
        rot = self.pointgroup_operations[nrot]
        rot_pos = np.dot(self.positions[atom] - self.positions[center], rot.T) + self.positions[center]
        diff = self.positions - rot_pos
        atom_ = np.where(np.all(np.abs(diff - np.rint(diff)) < self.symprec, axis=1))[0][0]
        return atom_

    def get_rotations_at_star(self, symmetries, center_atom, atom, symprec):
        map_operations = []
        rel_pos = self.positions - self.positions[center_atom]
        for j, nrot in enumerate(symmetries):
            r = self.pointgroup_operations[nrot]
            diff = np.dot(rel_pos[atom], r.T) - rel_pos[atom]
            if (abs(diff - np.rint(diff)) < symprec).all():
                map_operations.append(nrot)
        return np.intc(map_operations)
