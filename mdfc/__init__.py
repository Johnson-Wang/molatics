__author__ = 'xinjiang'

import sys
import numpy as np
from mdfc.symmetry import Symmetry
from phonopy.structure.cells import get_supercell, Primitive
from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors
from phonopy.file_IO import write_FORCE_CONSTANTS, write_force_constants_to_hdf5, read_force_constants_hdf5
from mdfc.file_IO import write_fc3_hdf5, write_fc2_hdf5, read_coefficient_from_hdf5, write_coefficient_to_hdf5
from force_constants import show_rotational_invariance,\
     set_translational_invariance, show_drift_force_constants
from mdfc.force_constants import ForceConstants
from mdfc.fcmath import gaussian
from realmd.information import timeit, print_error_message, warning
from copy import deepcopy
from phonopy.interface.vasp import write_vasp
import scipy.sparse as sparse
from realmd.memory_profiler import profile

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
        self._fc2_irred = None
        self._fc3 = None
        self._fc3_irred = None

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
        self._normalize_factor = 1.
        self._coeff2 = None
        self._coeff3 = None
        self._num_irred_fc2 = 0
        self._num_irred_fc3 = 0
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

    def run_fc1(self):
        symmetry = self.symmetry
        unique, indices = np.unique(symmetry.mapping, return_inverse=True)
        independents = []
        transforms = []
        for atom in unique:
            site_symmetry = symmetry.get_site_symmetry(atom)
            rots_cart = np.array([symmetry.get_cartesian_rotation(r) for r in site_symmetry])
            invariant_transforms = rots_cart - np.eye(3)
            CC, transform, independent = gaussian(invariant_transforms.reshape(-1, 3))
            transforms.append(transform)
            independents.append(independent)
        len_ind = [len(t) for t in independents]
        num_ind = sum(len_ind)

        coeff1 = np.zeros((self._num_atom, 3, num_ind), dtype='double')
        for i in np.arange(self._num_atom):
            rot_index = symmetry.rotations[symmetry.mapping_operations[i]]
            rot_inverse = symmetry.rot_inverse(rot_index)
            rot_inverse_cart = symmetry.get_cartesian_rotation(rot_inverse)
            atom_ = indices[i]
            trans = np.dot(rot_inverse_cart, transforms[atom_])
            ind_indices = np.arange(len_ind[atom_]) + sum(len_ind[:atom_])
            coeff1[i, :, ind_indices] = trans
        self._coeff1 = sparse.coo_matrix(coeff1.reshape(-1, num_ind))

    def run_fc2(self, is_read_coeff = False, is_write_coeff = False):
        if is_read_coeff:
            print "Reading harmonic coefficients from hdf5 file"
            self._coeff2 = read_coefficient_from_hdf5('fc2_coefficients.hdf5')
        else:
            self._fc.set_fc2_irreducible_elements()
            natom = self._num_atom
            tensor2 = self._symmetry.tensor2
            coeff_index = self._fc._coeff2
            ifc2_map = self._fc._ifc2_map
            irred_trans2 = self._fc._ifc2_trans
            num_irred_fc2 = self._fc._ifc2_trans.shape[-1]
            self._coeff2 = sparse.lil_matrix((natom * natom * 3 * 3, num_irred_fc2))
            for atom1 in np.arange(natom):
                coeff = tensor2[coeff_index[atom1]] # shape[natom, 9, 9]
                ipair = ifc2_map[atom1] # shape [natom]
                coeff_from_ele = np.einsum('ijk, ikm->ijm', coeff, irred_trans2[ipair]) # shape[natom, 9, nele]
                start = atom1 * natom * 9
                index_range = start + np.arange(natom * 9)
                coeff_tmp = coeff_from_ele.reshape(-1, num_irred_fc2)
                self._coeff2[index_range, :] = sparse.lil_matrix(coeff_tmp)
            if is_write_coeff:
                write_coefficient_to_hdf5(self._coeff2, 'fc2_coefficients.hdf5')
                print "Harmonic coefficients written to hdf5 file"
        self._coeff2 = self._coeff2.tocsr()
        self._num_irred_fc2 = self._coeff2.shape[-1]

    #@profile
    def init_disp_and_forces(self, coordinates=None, forces=None, is_normalize=True):
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
        if is_normalize:
            print "The displacements are normalized to have unit max displacement"
            # disp_norm = np.sqrt(np.average(self._displacements ** 2, axis=0)).max()
            disp_norm = np.average(np.abs(self._displacements), axis=0).max()
            self._displacements /= disp_norm
            self._normalize_factor = disp_norm
        self.show_residual_forces()

    def distribute_fc2(self):
        if self._fc2_irred is not None:
            fc2 = self._coeff2.dot(self._fc2_irred)
            self._fc2 = fc2.reshape(self._num_atom, self._num_atom, 3, 3)
        print "Force constants obtained from MD simulations"
        show_drift_force_constants(self.fc2)
        if self._is_trans_inv < 0:
            print "Coerced translational invariance mode, after which"
            set_translational_invariance(self.fc2)
            show_drift_force_constants(self.fc2)
        if self._is_hdf5:
            write_fc2_hdf5(self.fc2)
        else:
            write_FORCE_CONSTANTS(self._fc2, "FORCE_CONSTANTS_MDFC")

    def show_residual_forces(self, residual_forces=None):
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
        force_harm = np.dot(self._ddcs2, self._fc2_irred)
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

    #@profile
    def set_fc3_irreducible_components(self):
        if self._symmetry.tensor3 is None:
            self._symmetry.set_tensor3(True)
        fc = self._fc
        fc.set_third_independents()
        print "Number of irreduble triplets: %d" %len(fc._triplets)
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

    #@profile
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
        pinv = np.linalg.pinv(ddcs.reshape(-1, num_irred), rcond=1e-10)
        self._fc3_irred = 2 * np.dot(pinv, self._forces1.flatten())
        self._forces2 = self._forces1 - 1./2. * np.dot(ddcs, self._fc3_irred) # residual force after 3rd IFC

    def run_gradient_descent(self, steps=1000, lr=0.1, mu=1):
        if self._coeff2 is not None:
            num_irred_fc2 = self._coeff2.shape[-1]
            self._fc2_irred = np.zeros(num_irred_fc2, dtype='double')
        if self._coeff3 is not None:
            num_irred_fc3 = self._coeff3.shape[-1]
            self._fc3_irred = np.zeros(num_irred_fc3, dtype='double')
        for i in np.arange(steps):
            print "----%i th iteration"%(i+1)
            residual_force = self.forward_residual_force()
            print "Residual force (eV/A): %15.5f"%np.abs(residual_force).max()
            is_converge = True
            if self._fc2_irred is not None:
                delta_fc2 = self.backward_fc2(residual_force, learning_rate=lr)
                self._fc2_irred[:] += delta_fc2
                is_converge &= np.abs(delta_fc2).max() < 1e-3
            if self._fc3_irred is not None:
                delta_fc3 = self.backward_fc3(residual_force, learning_rate=lr)
                self._fc3_irred[:] += delta_fc3
                is_converge &= np.abs(delta_fc3).max() < 1e-3
            if is_converge:
                print "Fitting process reached convergence"
                break
        if not is_converge:
            print "Fitting process reached maximum steps without convergence"
        print_irred_fc2(self._fc._pairs, self._fc._ifc2_ele, self._fc2_irred)




    def forward_residual_force(self):
        num_step = len(self._displacements)
        forces = self._forces.reshape(num_step, -1)
        displacements = self._displacements.reshape(num_step, -1)
        forces_fc = np.zeros_like(forces)
        natom = self._num_atom
        if self._coeff2 is not None:
            fc2 = self._coeff2.dot(self._fc2_irred).reshape(natom, natom, 3, 3)
            fc2 = fc2.swapaxes(1, 2).reshape(natom * 3, natom * 3)
            forces_fc[:] += np.dot(displacements, fc2.T)
        if self._coeff3 is not None:
            fc3 = self._coeff3.dot(self._fc3_irred).reshape(natom, natom, natom, 3, 3, 3)
            fc3 = np.einsum('ijklmn->iljmkn', fc3).reshape(natom * 3, natom * 3, natom * 3)
            for i in np.arange(num_step):
                disp = self._displacements[i].flatten()
                forces_fc[i] += fc3.dot(disp).dot(disp)
        return self._forces - forces_fc.reshape(num_step, natom, 3) # get residual force

    def backward_fc2(self, residual_force, learning_rate=0.1):
        num_step = len(self._displacements)
        natom = self._num_atom
        delta = np.zeros_like(self._fc2_irred)
        for i in np.arange(num_step):
            disp = self._displacements[i]
            force = residual_force[i]
            fd = np.kron(force, disp).flatten()
            delta[:] += self._coeff2.T.dot(fd).T
        delta[:] /= num_step * natom * 3
        return delta * learning_rate

    def backward_fc3(self, residual_force, learning_rate=0.1):
        num_step = len(self._displacements)
        delta = np.zeros_like(self._fc3_irred)
        for i in np.arange(num_step):
            disp = self._displacements[i]
            force = residual_force[i]
            fd = np.kron(np.kron(force, disp), disp).flatten()
            delta[:] += self._coeff3.T.dot(fd).T
        delta[:] /= num_step * self._num_atom * 3
        return delta * learning_rate / 2









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

    #@profile
    def run_fc3(self, is_read_coeff = False, is_write_coeff = False):
        if is_read_coeff:
            print "Reading fc3 coefficients from hdf5 file"
            self._coeff3 = read_coefficient_from_hdf5('fc3_coefficients.hdf5')
        else:
            self.set_fc3_irreducible_components()
            natom = self._num_atom
            tensor3 = self._symmetry.tensor3
            coeff_index = self._fc._coeff3
            ifc3_map = self._fc._ifc3_map
            irred_trans3 = self._fc._ifc3_trans
            num_irred_fc3 = self._fc._ifc3_trans.shape[-1]
            nfreedom = natom * 3
            rows = []; columns = []; values = []
            for atom1 in np.arange(natom):
                coeff = tensor3[coeff_index[atom1]] #shape[natom, natom, 27, 27]
                ipair = ifc3_map[atom1] # shape[natom, natom, 27, nele]
                coeff_from_ele = np.einsum('ijkl, ijlm->ijkm', coeff, irred_trans3[ipair])
                start = atom1 * natom * natom * 27
                # index_range = start + np.arange(natom * natom * 27)
                coeff_tmp = coeff_from_ele.reshape(-1, num_irred_fc3)
                a, b = np.where(np.abs(coeff_tmp)>self._symprec)
                rows.append(a+start)
                columns.append(b)
                values.append(coeff_tmp[a,b])
            self._coeff3 = sparse.coo_matrix((np.hstack(values), (np.hstack(rows), np.hstack(columns))), shape=(nfreedom ** 3, num_irred_fc3))
            if is_write_coeff:
                write_coefficient_to_hdf5(self._coeff3, 'fc3_coefficients.hdf5')
                print "Fc3 coefficients written to hdf5 file"
        self._coeff3 = self._coeff3.tocsr()
        self._num_irred_fc3 = self._coeff3.shape[-1]

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
                                  self._symprec)
        self._pointgroup_operations = self._symmetry.get_pointgroup_operations()




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