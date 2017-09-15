__author__ = 'xinjiang'

import sys
import numpy as np
from phonopy.structure.symmetry import Symmetry
from phonopy.structure.cells import get_supercell, Primitive
from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors
from phonopy.file_IO import write_FORCE_CONSTANTS, write_force_constants_to_hdf5, read_force_constants_hdf5
from mdfc.file_IO import write_fc3_hdf5, write_fc2_hdf5
from force_constants import show_rotational_invariance,\
     set_translational_invariance, show_drift_force_constants
# from mdfc.fc2 import get_irreducible_components2, get_fc2_least_irreducible_components, get_disp_coefficient
# from mdfc.fc3 import show_drift_fc3 #, get_fc3_irreducible_components
from mdfc.force_constants import ForceConstants
from realmd.information import timeit, print_error_message, warning
from copy import deepcopy
from phonopy.interface.vasp import write_vasp
# from realmd.memory_profiler import profile


def print_irred_fc2(pairs_reduced, ifc2_ele, irred_fc2):
    tensor = {0:'xx', 1:'xy', 2:'xz', 3:'yx', 4:'yy', 5:'yz', 6:'zx', 7:'zy',8:'zz'}
    print "Irreducible fc2 elements: (eV/A^2)"
    for i, (ele, fc2) in enumerate(sorted(zip(ifc2_ele, irred_fc2))):
        direct = tensor[ele % 9]
        pair = pairs_reduced[ele // 9]
        if i>1 and i%5 == 0:
            print
        print "{:>3d}-{:<3d}({:s}):{:12.4e}".format(pair[0], pair[1], direct, fc2),
    print
    sys.stdout.flush()

def print_irred_fc3(triplets_reduced, ifc3_ele, irred_fc3):
    tensor = {0:'xxx', 1:'xxy', 2:'xxz', 3:'xyx', 4:'xyy', 5:'xyz', 6:'xzx', 7:'xzy',8:'xzz',
              9:'yxx', 10:'yxy', 11:'yxz', 12:'yyx', 13:'yyy', 14:'yyz', 15:'yzx', 16:'yzy',17:'yzz',
              18:'zxx', 19:'zxy', 20:'zxz', 21:'zyx', 22:'zyy', 23:'zyz', 24:'zzx', 25:'zzy',26:'zzz'}
    print "Irreducible fc3 elements: (eV/A^3)"
    for i, (ele, fc3) in enumerate(sorted(zip(ifc3_ele, irred_fc3))):
        direct = tensor[ele % 27]
        triplet = triplets_reduced[ele // 27]
        if i>1 and i%5 == 0:
            print
        print "{:>3d}{:-^5d}{:<3d}({:s}):{:12.4e}".format(triplet[0], triplet[1], triplet[2], direct, fc3),
        # print "%3d-%3d-%-3d(%s): %11.4e |" %(triplet[0], triplet[1], triplet[2], direct, fc3),
    print
    sys.stdout.flush()

class MolecularDynamicsForceConstant:
    def __init__(self,
                 unitcell,
                 supercell_matrix,
                 cutoff_radius=None, # unit: Angstrom
                 cutoff_force=1e-8,
                 cutoff_disp=None,
                 factor=1,
                 symprec=1e-5,
                 divide=1,
                 count=1,
                 is_symmetry=True,
                 is_translational_invariance=False,
                 is_rotational_invariance=False,
                 is_disperse=False,
                 precision=1e-8,
                 log_level=0,
                 is_hdf5=False):
        self._symprec = symprec
        self._unitcell = unitcell
        self._supercell_matrix = supercell_matrix
        self._factor = factor
        self._is_symmetry = is_symmetry
        self._log_level = log_level
        self._supercell = None
        self._set_supercell()
        self._symmetry = None
        self._set_symmetry()
        self._fc2 = None
        self.set_cutoffs(cutoff_radius)
        self._cutoff_force=cutoff_force
        self._cutoff_disp = cutoff_disp
        self._is_rot_inv = is_rotational_invariance
        self._is_trans_inv = is_translational_invariance
        self._precision = precision
        self._is_hdf5 = is_hdf5
        self._count = count
        self._positions = self.supercell.get_scaled_positions()
        self._lattice = self.supercell.get_cell()
        self._num_atom = len(self._positions)
        self.ifc2_map = None
        self._coeff2 = None
        self._ind_atoms2 = None
        self._num_irred_fc2 = None
        self.map_atoms2 = None
        self.map_operations2 = None
        self._coeff2 = None
        self._converge2 = False
        self._fc = ForceConstants(self.supercell,
                                  self.primitive,
                                  self.symmetry,
                                  is_disperse=is_disperse,
                                  cutoff=self._cutoff,
                                  precision = precision)
        self._step = 0
        self._forces1 = None
        self._forces2 = None
        self._fc3 = None
        self._fc3_irred = None


    def __iter__(self):
        return self


    def get_primitive(self):
        return self._primitive
    primitive = property(get_primitive)

    def set_primitive(self, primitive):
        self._primitive = primitive

    def get_unitcell(self):
        return self._unitcell
    unitcell = property(get_unitcell)

    def get_supercell(self):
        return self._supercell
    supercell = property(get_supercell)

    def set_supercell(self, supercell):
        self._supercell = supercell

    def get_symmetry(self):
        return self._symmetry
    symmetry = property(get_symmetry)

    def get_unit_conversion_factor(self):
        return self._factor
    unit_conversion_factor = property(get_unit_conversion_factor)

    def get_fc(self):
        return self._fc
    fc = property(get_fc)

    def get_fc2(self):
        return self._fc2
    fc2 = property(get_fc2)


    def run_fc2(self):
        self._fc.set_fc2_irreducible_elements(is_trans_inv=self._is_trans_inv, is_rot_inv=self._is_rot_inv)
        self._coeff2 = self._fc._coeff2
        self.ifc2_map = self._fc._ifc2_map
        self.irred_trans = self._fc._ifc2_trans
        self._num_irred_fc2 = len(self._fc._ifc2_ele)
        # self.set_disp_and_forces()
        for i in self:
            pass


    def next(self):
        print "The %dth iteration in finding the equilibrium position"%(self._step + 1)
        self.set_disp_and_forces()
        self.distribute_fc2()
        self.show_residual_forces_fc2()
        print_irred_fc2(self._fc._pairs_reduced, self._fc._ifc2_ele, self.irred_fc2)
        self._step += 1
        if self._converge2:
            print "Equilibrium position found. "
            raise StopIteration
        elif self._step == self._count:
            print "Iteration terminated due to the counting limit"
            raise StopIteration
        else:
            self.predict_new_positions()
            cell = deepcopy(self.supercell)
            cell.set_positions(self._pos_equi)
            write_vasp("POSCAR_FC2", cell, direct=False)


    def init_disp_and_forces(self, coordinates=None, forces=None):
        self._forces = - forces # the negative sign enforces F = Phi .dot. U
        self._resi_force = np.average(forces, axis=0)
        self._resi_force_abs = np.zeros_like(self._resi_force)
        # self._pos_equi = self.supercell.get_positions()
        print "Initial equilibrium positions are set as the average positions"
        self._pos_equi = np.average(coordinates, axis=0)
        self._displacements = coordinates - self._pos_equi
        assert (np.abs(self._displacements).max(axis=(0,1)) <  np.sqrt(np.sum(self.supercell.get_cell() ** 2, axis=0)) / 2).all(),\
            "The coordinates should not be folded, that is, once an atom goes beyond the box\
               it should not be dragged to the opposite corner"
        self.show_residual_forces()

    @timeit
    def set_disp_and_forces(self, lang="C"):
        num_step = len(self._displacements)
        # self.irred_fc = np.zeros(self._num_irred_fc2, dtype="double")
        forces = self._forces[:]
        ddcs2 = np.zeros((num_step, self._num_atom, 3,  self._num_irred_fc2), dtype="double")
        if lang == "C":
            import _mdfc as mdfc
            mdfc.rearrange_disp_fc2(ddcs2,
                                    self._displacements,
                                    self._coeff2.astype("double").copy(),
                                    self.irred_trans.astype("double").copy(),
                                    self.ifc2_map.astype("intc").copy(),
                                    1e-6)
        else:
            ddcs2 = np.zeros((num_step, self._num_atom, 3,  self._num_irred_fc2), dtype="double")
            for atom1 in range(self._num_atom):
                for atom2 in range(self._num_atom):
                    disp = self._displacements[:, atom2]
                    coeff = np.dot(self._coeff2[atom1, atom2],
                                   self.irred_trans[self.ifc2_map[atom1, atom2]]).reshape(3,3,-1)
                    ddcs2[:, atom1] += np.einsum("abn, Nb -> Nan", coeff, disp)# i: md steps; j; natom; k: 3

        self._ddcs2 = ddcs2
        ddcs2 = ddcs2.reshape((-1, self._num_irred_fc2))
        try:
            raise ImportError
            # The lapacke pseudo inverse is surprisingly much slower yet more memory consuming than numpy
            import _mdfc
            print "Pseudo-inverse is realized by lapack package"
            irred_uA_pinv = np.zeros(ddcs2.shape[::-1], dtype="double")
            _mdfc.pinv(ddcs2, irred_uA_pinv, 1e-5)
        except ImportError:
            print "Numpy is used to to realize Moore-Penrose pseudo inverse"
            irred_uA_pinv = np.linalg.pinv(ddcs2)
        self.irred_fc2 = np.dot(irred_uA_pinv, forces.flatten())

    def distribute_fc2(self):
        self._fc2 = np.zeros((self._num_atom, self._num_atom, 3, 3))
        for atom1 in np.arange(self._num_atom):
            for atom2 in np.arange(self._num_atom):
                trans = np.dot(self._coeff2[atom1, atom2], self.irred_trans[self.ifc2_map[atom1, atom2]])
                self._fc2[atom1, atom2] = np.dot(trans, self.irred_fc2).reshape(3,3)
        print "Force constants obtained from MD simulations"
        # show_rotational_invariance(self.force_constants, self.supercell, self.primitive)
        show_drift_force_constants(self.fc2)
        if self._is_trans_inv < 0:
            print "Coerced translational invariance mode, after which"
            set_translational_invariance(self.fc2)
            show_drift_force_constants(self.fc2)
        if self._is_hdf5:
            write_fc2_hdf5(self.fc2)
        else:
            write_FORCE_CONSTANTS(self._fc2, "FORCE_CONSTANTS_MDFC")

    def show_residual_forces(self):
        resi_force_abs = np.sqrt(np.average(self._forces ** 2, axis=0))
        disp_abs = np.sqrt(np.average(self._displacements ** 2, axis=0))
        if self._log_level >= 2:
            print "####Root mean square average amplitude of residual forces on each atom (eV/Angstrom)"
            for i in range(self._num_atom):
                print "Atom %3d: %7.4e %7.4e %7.4ef" %(i, resi_force_abs[i,0], resi_force_abs[i,1], resi_force_abs[1,2])
            print "####Root mean square average amplitude of displacement (Angstrom)"
            for i in range(self._num_atom):
                print "Atom %3d: %7.4e %7.4e %7.4ef" %(i, disp_abs[i,0], disp_abs[i,1], disp_abs[1,2])
        elif self._log_level == 1:
            print "####Root mean square average amplitude of residual force (eV/Angstrom)"
            print "Total: %7.4e %7.4e %7.4e" %tuple(np.average(resi_force_abs, axis=0))
            print "####Root mean square average amplitude of displacement (Angstrom)"
            print "Total: %7.4e %7.4e %7.4e" %tuple(np.average(disp_abs, axis=0))
        sys.stdout.flush()


    def show_residual_forces_fc2(self):
        # force_harm = np.tensordot(self._displacements, self.fc2, axes=((1, 2), (1, 3)))
        force_harm = np.dot(self._ddcs2, self.irred_fc2)
        resi_force = self._forces - force_harm
        self._forces1 = resi_force[:]
        resi_force_abs = np.sqrt(np.average(resi_force ** 2, axis=0))
        disp_abs = np.sqrt(np.average(self._displacements ** 2, axis=0))
        if self._cutoff_disp is not None:
            weight = np.exp(-resi_force ** 2 / self._cutoff_disp ** 2 /  2)
            # weight = np.exp(-self._displacements ** 2 / self._cutoff_disp ** 2 /  2)
            self._resi_force = np.average(resi_force, axis=0, weights=weight)
            print "Minimum weight: %d" %np.sum(weight, axis=0).min()
        else:
            self._resi_force = np.average(resi_force, axis=0)
        if self._log_level >= 2:
            print "####Root mean square average amplitude of residual forces on each atom (eV/Angstrom)"
            for i in range(self._num_atom):
                print "Atom %3d: %7.4e %7.4e %7.4ef" %(i, resi_force_abs[i,0], resi_force_abs[i,1], resi_force_abs[1,2])
            print "####Arithmetic average amplitude of residual forces on each atom (eV/Ang)"
            resi_force_atom = np.average(resi_force, axis=0)
            for i in range(self._num_atom):
                print "Atom %3d: %7.4e %7.4e %7.4ef" %(i, resi_force_atom[i,0], resi_force_atom[i,1], resi_force_atom[i,2])
            print "####Root mean square average amplitude of displacement (Angstrom)"
            for i in range(self._num_atom):
                print "Atom %3d: %7.4e %7.4e %7.4ef" %(i, disp_abs[i,0], disp_abs[i,1], disp_abs[1,2])
        elif self._log_level == 1:
            print "####Root mean square average amplitude of residual force (eV/Ang)"
            print "Total: %7.4e %7.4e %7.4e" %tuple(np.average(resi_force_abs, axis=0))
            print "####Arithmetic average amplitude of residual force (eV/Ang)"
            print "Total: %7.4e %7.4e %7.4e" %tuple(np.sqrt(np.average(self._resi_force ** 2, axis=0)))
            print "####Root mean square average amplitude of displacement (Angstrom)"
            print "Total: %7.4e %7.4e %7.4e" %tuple(np.average(disp_abs, axis=0))
        sys.stdout.flush()
        if (np.sqrt(np.average(self._resi_force ** 2, axis=0)).max()) < self._cutoff_force:
            self._converge2 = True
        self._resi_force_abs = resi_force_abs

    def predict_new_positions(self):
        fc = np.swapaxes(self.fc2, 1, 2).reshape(self._num_atom * 3, self._num_atom * 3)
        phi_inv = np.linalg.pinv(fc, rcond=1e-10)
        forces = self._resi_force.flatten()
        disp = np.dot(phi_inv, forces).reshape(self._num_atom, 3)
        self._pos_equi -= disp
        self._displacements += disp

    @timeit
    def set_fc3_irreducible_components(self):
        fc = self._fc
        fc.set_third_independents()
        print "Number of 3rd IFC: %d" %(27 * len(fc._triplets))
        fc.get_irreducible_fc3s_with_permute()
        print "Permutation reduces 3rd IFC to %d"%(27 * len(fc._triplets_reduced))
        fc.get_irreducible_fc3_components_with_spg()
        print "spg invariance reduces 3rd IFC to %d"%(len(fc._ifc3_ele))
        print "Calculating fc3 coefficient..."
        fc.get_fc3_coefficients()
        if self._is_trans_inv:
            fc.get_fc3_translational_invariance()
            print "translational invariance reduces 3rd IFC to %d"%(len(fc._ifc3_ele))
        if self._is_rot_inv and self.fc2 is not None:
            fc.get_fc3_rotational_invariance(self._fc2)
            print "rotational invariance reduces 3rd IFC to %d"%(len(fc._ifc3_ele))
        self._num_irred_fc3 = len(fc._ifc3_ele)

    @timeit
    def calculate_irreducible_fc3(self):
        fc = self._fc
        coeff = fc._coeff3
        ifcmap = fc._ifc3_map
        trans = fc._ifc3_trans
        num_irred = trans.shape[-1]
        num_step = len(self._displacements)
        print "Calculating fc3..."
        sys.stdout.flush()
        import _mdfc as mdfc
        # ddcs3 = np.zeros((num_step * self._num_atom * 3, num_irred), dtype="double")
        ddcs = np.zeros((num_step, self._num_atom, 3, num_irred), dtype="double") #displacements  rearrangement as the coefficients of fc3
        mdfc.rearrange_disp_fc3(ddcs,
                                self._displacements,
                                coeff,
                                trans,
                                ifcmap,
                                1e-6)
        # self._ddcs = np.contatenate((self._ddcs, ddcs3 / 2), axis=2)
        # for atom1 in np.arange(self._num_atom):
        #     for atom2 in np.arange(self._num_atom):
        #         disp2 = self._displacements[:,atom2]
        #         for atom3 in np.arange(atom2, self._num_atom):
        #             disp3 = self._displacements[:,atom3]
        #             num_triplet = ifcmap[atom1, atom2, atom3]
        #             coeff_temp = np.dot(coeff[atom1, atom2, atom3], trans[num_triplet]).reshape(3,3,3, num_irred)
        #             sum_temp  = np.einsum("ijkn, Nj, Nk -> Nin", coeff_temp, disp2, disp3)
        #             if atom2 == atom3:
        #                 sum_temp *= 2
        #             ddcs[:,atom1] += sum_temp
        pinv = np.linalg.pinv(ddcs.reshape(-1, num_irred), rcond=1e-10)
        self._fc3_irred = 2 * np.dot(pinv, self._forces1.flatten())
        self._forces2 = self._forces1 - 1./2. * np.dot(ddcs, self._fc3_irred) # residual force after 3rd IFC




    # def calculate_fcs(self):
    #     self._fc_irred = np.zeros((self._divide, self._ddcs.shape[-1]), dtype="double")
    #     for i in range(self._divide):
    #         ddcs = self._ddcs[i]
    #         pinv = np.linalg.pinv(ddcs)
    #         fcs = np.dot(pinv, self._forces.flatten())
    #         self._fc_irred[i] = fcs





    @timeit
    def distribute_fc3(self):
        fc = self._fc
        coeff = fc._coeff3
        ifcmap = fc._ifc3_map
        trans = fc._ifc3_trans
        num_irred = trans.shape[-1]
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
        show_drift_fc3(self._fc3)

    def show_residual_fc3(self):
        resi_force_abs = np.sqrt(np.average(self._forces2 ** 2, axis=0))
        resi_force = np.average(self._forces2, axis=0)
        print "3rd order IFC sucessfully obtained."
        if self._log_level >= 2:
            print "####Root mean square average amplitude of residual forces on each atom (eV/Ang)"
            for i in range(self._num_atom):
                print "Atom %3d: %7.4e %7.4e %7.4ef" %(i, resi_force_abs[i,0], resi_force_abs[i,1], resi_force_abs[1,2])
            print "####Arithmetic average amplitude of residual forces on each atom (eV/Ang)"
            for i in range(self._num_atom):
                print "Atom %3d: %7.4e %7.4e %7.4ef" %(i, resi_force[i,0], resi_force[i,1], resi_force[1,2])
        elif self._log_level == 1:
            print "####Root mean square average amplitude of residual force (eV/Ang)"
            print "Total: %7.4e %7.4e %7.4e" %tuple(np.average(resi_force_abs, axis=0))
            print "####Arithmetic average amplitude of residual force (eV/Ang)"
            print "Total: %7.4e %7.4e %7.4e" %tuple(np.average(resi_force, axis=0))
        sys.stdout.flush()

    @timeit
    def run_fc3(self):
        self.set_fc3_irreducible_components()
        self.calculate_irreducible_fc3()
        self.show_residual_fc3()
        self.distribute_fc3()
        print_irred_fc3(self._fc._triplets_reduced, self._fc._ifc3_ele, self._fc3_irred)
        write_fc3_hdf5(self._fc3)
        # for atom1 in np.arange(self._num_atom):
        #     for atom2 in np.arange(self._num_atom):
        #         disp2 = self._displacements[:,atom2]
        #         for atom3 in np.arange(self._num_atom):
        #             disp3 = self._displacements[:,atom3]
        #             num_triplet = ifcmap[atom1, atom2, atom3]
        #             coeff_temp = np.dot(coeff[atom1, atom2, atom3], trans[num_triplet]).reshape(3,3,3, num_irred)
        #             ddcs[:,atom1] += np.einsum("ijkn, Nj, Nk -> Nin", coeff_temp, disp2, disp3)




    def set_cutoffs(self, cutoff_radius=None):
        species = []
        for symbol in self.unitcell.get_chemical_symbols():
            if symbol not in species:
                species.append(symbol)
        self._cutoff = Cutoff(species, cutoff_radius)
        self._cutoff.set_cell(self.supercell, symprec=self._symprec)


    def _create_force_matrix(self,
                             sets_of_forces,
                             site_sym_cart,
                             rot_map_syms):
        force_matrix = []
        for i in range(self._num_atom):
            for forces in sets_of_forces:
                for f, ssym_c in zip(
                    forces[rot_map_syms[:, i]], site_sym_cart):
                    force_matrix.append(np.dot(ssym_c, f))
        return np.reshape(force_matrix, (self._num_atom, -1, 3))

    def _create_displacement_matrix(self,
                                    disps,
                                    site_sym_cart):
        rot_disps = []
        for u in disps:
            for ssym_c in site_sym_cart:
                Su = np.dot(ssym_c, u)
                rot_disps.append(Su)
        return np.array(rot_disps, dtype='double')

    def _set_supercell(self):
        supercell = get_supercell(self._unitcell,
                                        self._supercell_matrix,
                                        self._symprec)
        self.set_supercell(supercell)
        self._primitive = Primitive(supercell,
                                    np.linalg.inv(self._supercell_matrix),
                                    self._symprec)

    def _set_symmetry(self):
        self._symmetry = Symmetry(self._supercell,
                                  self._symprec,
                                  self._is_symmetry)



    def set_force_constants(self, force_constants):
        self._fc2 = force_constants

    def set_force_sets(self, sets_of_forces_objects):
        self._set_of_forces_objects = sets_of_forces_objects


class Cutoff():
    def __init__(self, species, cut_radius):
        self._cell = None
        self._pair_distances = None
        n = len(species)
        self._cut_radius = None
        if cut_radius is not None:
            if len(cut_radius) == 1:
                self._cut_radius_species = {species[i]:cut_radius[0] for i in range(n)}
            elif len(cut_radius) == n:
                self._cut_radius_species = {species[i]:cut_radius[i] for i in range(n)}
            else:
                print_error_message("Cutoff radius number %d not equal the number of species %d!" %(len(cut_radius), n))
            print "Cutoff radius of atoms (A)"
            for i in range(n):
                print "%3s: %5.2f;" %(species[i], self._cut_radius_species[species[i]]),
            print
        else:
            self._cut_radius_species = None

    def set_cell(self, cell, symprec = 1e-5):
        self._cell = cell
        self._symprec = symprec
        num_atom = self._cell.get_number_of_atoms()
        chemical_symbols = self._cell.get_chemical_symbols()
        if self._cut_radius_species is not None:
            self._cut_radius = np.zeros(num_atom, dtype='double')
            for i in range(num_atom):
                self._cut_radius[i] = self._cut_radius_species[chemical_symbols[i]]
        self._pair_distances = None

    def get_cutoff_radius(self):
        return self._cut_radius_species

    def set_pair_distances(self):
        num_atom = self._cell.get_number_of_atoms()
        lattice = self._cell.get_cell()
        min_distances = np.zeros((num_atom, num_atom), dtype='double')
        for i in range(num_atom): # run in cell
            for j in range(i): # run in primitive
                min_distances[i, j] = np.linalg.norm(np.dot(
                        get_equivalent_smallest_vectors(
                            i, j, self._cell, lattice, self._symprec)[0], lattice))
        self._pair_distances = (min_distances + min_distances.T) / 2.

    def get_pair_inclusion(self, pairs=None):
        lattice = self._cell.get_cell()
        include_pair = np.ones(len(pairs), dtype=bool)
        if self._cut_radius_species is not None:
            for i, (a1, a2) in enumerate(pairs):
                distance = \
                    np.linalg.norm(np.dot(get_equivalent_smallest_vectors(
                            a2, a1, self._cell, lattice, self._symprec)[0], lattice))
                if distance > self._cut_radius[a1] + self._cut_radius[a2]:
                    include_pair[i] = False
        return include_pair

    def get_triplet_inclusion(self, triplets=None):
        lattice = self._cell.get_cell()
        include_triplet = np.ones(len(triplets), dtype=bool)
        for i, (a1, a2, a3) in enumerate(triplets):
            d12 = \
                np.linalg.norm(np.dot(get_equivalent_smallest_vectors(
                        a2, a1, self._cell, lattice, self._symprec)[0], lattice))
            d23 = \
                np.linalg.norm(np.dot(get_equivalent_smallest_vectors(
                        a3, a2, self._cell, lattice, self._symprec)[0], lattice))
            d13 = \
                np.linalg.norm(np.dot(get_equivalent_smallest_vectors(
                        a3, a1, self._cell, lattice, self._symprec)[0], lattice))
            if d12 > self._cut_radius[a1] + self._cut_radius[a2] and\
                d23 > self._cut_radius[a2] + self._cut_radius[a3] and\
                d13 > self._cut_radius[a1] + self._cut_radius[a3]:
                include_triplet[i] = False
        return include_triplet