import spglib
import numpy as np
from phonopy.structure.cells import Primitive
class Symmetry():
    def __init__(self, cell, symprec=1e-6):
        self.cell = cell
        self.positions = self.cell.get_scaled_positions()
        self.symprec = symprec
        self.rot_inv = None
        self.rot_mult = None
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

    def get_atom_mapping_under_sitesymmetry(self, center_atom, atom):
        """atom is mapped to a lower(or equal) index under the sitesymmetry at center_atom"""
        symmetries = self.get_site_symmetry(center_atom)
        positions = self.positions - self.positions[center_atom]
        if len(symmetries) == 1: # speeding up the case of no symmetries
            return atom, 0
        p = positions[atom]
        pos2 = self.pointgroup_operations[symmetries].dot(p)
        diff = pos2[np.newaxis] - positions[:atom+1, np.newaxis]
        diff -= np.rint(diff)
        return zip(*np.where(np.all(np.abs(diff) < self.symprec, axis=-1)))[0]
        # positions[:atom+1]
        # for j in range(atom+1):
        #     diff = self.pointgroup_operations[symmetries].dot(p) - positions[j]
        #     diff -= np.rint(diff)
        #     np.all(np.abs(diff) < self.symprec, axis=1)
        #     if np.all(np.abs(diff) < self.symprec, axis=1).any():
        #         return j, np.where()
        #     for k in symmetries:
        #         r = self.pointgroup_operations[k]
        #         if np.allclose(np.dot(r, p), positions[j], atol=self.symprec):
        #             return j, k


    def set_multiplication_table(self):
        num_rot = len(self.pointgroup_operations)
        self.rot_mult = np.zeros((num_rot, num_rot), dtype="intc")
        self.rot_inv = np.zeros(num_rot, dtype='intc')
        for i, pg1 in enumerate(self.pointgroup_operations):
            for j, pg2 in enumerate(self.pointgroup_operations):
                pg = np.dot(pg1, pg2)
                k = np.all(pg==self.pointgroup_operations, axis=(1,2)).argmax()
                # for k, pg3 in enumerate(self.pointgroup_operations):
                #     if (pg == pg3).all():
                #         break
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

# def get_atom_sent_by_operation(orig_atom, positions, r, t, symprec=1e-5):
#     rot_pos = np.dot(positions[orig_atom], r.T) + t
#     diff = positions - rot_pos
#     atom2 = np.where(np.all(np.abs(diff - np.rint(diff)) < symprec, axis=1))[0][0]
#     return atom2
#
#
# def get_atom_sent_by_site_sym(atom, center, positions, rot, symprec=1e-5):
#     rot_pos = np.dot(positions[atom] - positions[center], rot.T) + positions[center]
#     diff = positions - rot_pos
#     atom2 = np.where(np.all(np.abs(diff - np.rint(diff)) < symprec, axis=1))[0][0]
#     return atom2


# def get_operations_at_star(operations, positions, atom, mappings, symprec):
#     star = mappings[atom]
#     map_operations = []
#     for j, (r, t) in enumerate(
#             zip(operations['rotations'], operations['translations'])):
#         diff = np.dot(positions[atom], r.T) + t - positions[star]
#         if (abs(diff - np.rint(diff)) < symprec).all():
#             map_operations.append(j)
#     return map_operations


# def get_all_operations_at_star(rotations, translations, positions, atom, mappings, symprec):
#     star = mappings[atom]
#     map_operations = []
#     for j, (r, t) in enumerate(
#             zip(rotations, translations)):
#         diff = np.dot(positions[atom], r.T) + t - positions[star]
#         if (abs(diff - np.rint(diff)) < symprec).all():
#             map_operations.append(j)
#     return map_operations
#
#
# def get_rotations_at_star(site_symmetries, positions, center_atom, atom, mappings, symprec):
#     star = mappings[atom]
#     map_operations = []
#     rel_pos = positions - positions[center_atom]
#     for j, r in enumerate(site_symmetries):
#         diff = np.dot(rel_pos[atom], r.T) - rel_pos[star]
#         if (abs(diff - np.rint(diff)) < symprec).all():
#             map_operations.append(j)
#     return site_symmetries[map_operations]


# def get_next_atom(center, site_symmetry, positions, symprec=1e-5):
#     """next_atom: a dict which should at least contain the key 'atom_number'
#     site_symmetry: the site_symmetry at the center atom (atom_number)"""
#     next_atom = {}
#     next_atom['atom_number'] = center
#     rela_pos = positions - positions[center]
#     map_atoms = np.arange(len(positions))
#     map_ops = np.zeros(len(positions), dtype=int)
#     next_atom['site_symmetry'] = site_symmetry
#     for i, p in enumerate(rela_pos):
#         is_found = False
#         for j in range(i):
#             for k,r in enumerate(site_symmetry):
#                 diff = np.dot(p, r.T) - rela_pos[j]
#                 diff -= np.rint(diff)
#                 if (abs(diff) < symprec).all():
#                     map_atoms[i] = j
#                     map_ops[i] = k
#                     is_found = True
#                     break
#             if is_found:
#                 break
#     next_atom['mapping'] =map_atoms
#     next_atom['mapping_operation'] = map_ops
#     next_atom['independent_atoms'] = np.unique(map_atoms)
#     next_atom['next_atoms'] = [{"atom_number":a} for a in np.unique(map_atoms)]
#     return next_atom