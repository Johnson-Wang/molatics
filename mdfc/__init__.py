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
from mdfc.fc2 import get_irreducible_components2, get_disp_coefficient, get_fc2_least_irreducible_components
from mdfc.fc3 import  get_fc3_irreducible_components, show_drift_fc3
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
                 cutoff_pair=None,# unit: Angstrom
                 cutoff_triplet=None,
                 cutoff_force=1e-8,
                 cutoff_disp=None,
                 factor=1,
                 symprec=1e-5,
                 divide=1,
                 count=1,
                 is_symmetry=True,
                 is_translational_invariance=False,
                 is_rotational_invariance=False,
                 is_weighted=False,
                 log_level=0,
                 is_hdf5=True):
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
        self._primitive = None
        self._fc2 = None
        self.set_cutoffs(cutoff_pair, cutoff_triplet)
        self._cutoff_force=cutoff_force
        self._cutoff_disp = cutoff_disp
        self._is_rot_inv = is_rotational_invariance
        self._is_trans_inv = is_translational_invariance
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
        self._fc = ForceConstants(self.supercell, self.symmetry)
        self._is_weighted=is_weighted
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

    def run_fc2(self):
        fc = self._fc
        fc.set_first_independents()
        print "Under the system symmetry"
        print "Number of first independent atoms: %4d" % fc._ind1['natoms']
        fc.set_second_independents(pair_included=self._cutoff.get_pair_inclusion())
        print "Number of second independent atoms %4d" %np.sum(fc._ind2['natoms'])
        # if not self._cutoff.get_pair_inclusion().all():
        #     print "Number of second independent atoms reduces to %d after cutoff" %fc._ind2['included'].sum()
        fc.get_irreducible_fc2s_with_permute()
    

        fc.get_fc2_coefficients()
        if self._cutoff.get_cutoff_pair() is not None:
            pair_inclusion = self._cutoff.get_pair_inclusion()
            fc.set_pair_reduced_included(pair_inclusion)
        else:
            fc.set_pair_reduced_included()
        print "Permutation symmetry reduces number of irreducible pairs from %4d to %4d"\
              %(np.sum(fc._ind2['natoms']), len(fc._pairs_reduced))
        if self._cutoff.get_cutoff_pair() is not None:
            print "The artificial cutoff reduces number of irreducible pairs from %4d to %4d"\
                  %(len(fc._pairs_reduced), np.sum(fc._is_pairs_included))
        print "Calculating transformation coefficients..."
        print "Number of independent fc2 components: %d" %(np.sum(fc._is_pairs_included)*9)
        fc.get_irreducible_fc2_components_with_spg()
        print "Point group invariance reduces independent fc2 components to %d" %(len(fc._ifc2_ele))
        sys.stdout.flush()
        if self._is_trans_inv:
            fc.get_fc2_translational_invariance()
            print "Translational invariance further reduces independent fc2 components to %d" %len(fc._ifc2_ele)
        if self._is_rot_inv:
            fc.get_fc2_rotational_invariance(self.unitcell)
            print "Rotational invariance further reduces independent fc2 components to %d" %len(fc._ifc2_ele)
        print "Independent fc2 components calculation completed"
        sys.stdout.flush()
        #
        # fc2 = read_force_constants_hdf5("fc2.hdf5")
        # fc2_reduced = np.array([fc2[pai] for pai in fc._pairs_reduced])
        # fc2p = fc2_reduced.flatten()[fc._ifc2_ele]
        # pair = (3,1)
        # trans = np.dot(fc._coeff2[pair], fc._ifc2_trans[fc._ifc2_map[pair]])
        # print np.dot(trans, fc2p).reshape(3,3)
        # print fc2[pair]
        # print np.dot(fc._coeff2[pair], fc2_reduced[fc._ifc2_map[pair]].flatten()).reshape(3,3)

        self._coeff2 = fc._coeff2
        self.ifc2_map = fc._ifc2_map
        self.irred_trans = fc._ifc2_trans
        self._num_irred_fc2 = len(fc._ifc2_ele)
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
        self._pos_equi = self.supercell.get_positions()
        print "Initial equilibrium positions are set as the average positions"
        # self._pos_equi = np.average(coordinates, axis=0)
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
        weight = np.ones_like(self._fc._is_pairs_included, dtype="double")
        if lang == "C":
            import _mdfc as mdfc
            mdfc.rearrange_disp_fc2(ddcs2,
                                    self._displacements,
                                    self._coeff2.astype("double").copy(),
                                    self._fc._is_pairs_included.copy().astype("int8"),
                                    self.irred_trans.astype("double").copy(),
                                    self.ifc2_map.astype("intc").copy(),
                                    1e-6)
        else:
            pair_inclusion = self._fc._is_pairs_included
            ddcs2 = np.zeros((num_step, self._num_atom, 3,  self._num_irred_fc2), dtype="double")
            for atom1 in range(self._num_atom):
                for atom2 in range(self._num_atom):
                    map_pair = self.ifc2_map[atom1, atom2]
                    if pair_inclusion[map_pair]:
                        continue
                    disp = self._displacements[:, atom2]
                    coeff = np.dot(self._coeff2[atom1, atom2],
                                   self.irred_trans[map_pair]).reshape(3,3,-1)
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
        # force_harm = np.einsum("Nbj, abij -> Nai",self._displacements, self.fc2)
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
        # dds = get_fourth_order_displacements(self.supercell, self.symmetry, is_plusminus=False, is_diagonal=False)
        # coeff, ifcmap, trans = get_fc3_irreducible_components(self.supercell, self.symmetry, self.fc2)
        fc = self._fc

        # import h5py
        # f = h5py.File("fc3.hdf5")
        # fc3 = f['fc3'][:]
        # f.close()



        fc.set_third_independents()
        # fc3_triplets = np.array([fc3[tt] for tt in fc._triplets]) # delete
        print "Number of triplets in calculating 3rd IFC: %d" %len(fc._triplets)
        fc.get_irreducible_fc3s_with_permute()
        print "Permutation reduces triplets to %d"%len(fc._triplets_reduced)
        print "Calculating fc3 coefficient..."
        fc.get_fc3_coefficients()
        if self._cutoff.get_cutoff_triplet() is not None:
            print "Setting cutoff on fc3..."
            triplets_inclusion = self._cutoff.get_triplet_inclusion(triplets=fc._triplets_reduced)
            fc.set_triplet_reduced_included(triplets_inclusion)
            print "Cutoff on FC3 reduces the number of triplets further to %d"%np.sum(fc._is_triplets_included)
        else:
            fc.set_triplet_reduced_included()
        # fc3_ir_triplets = np.array([fc3[tt] for tt in fc._triplets_reduced]) # delete
        # fc_in = np.array([fc3[fc._triplets_reduced[ppp]] for ppp in np.where(fc._is_triplets_included)[0]])
        # fc_ni = np.array([fc3[fc._triplets_reduced[ppp]] for ppp in np.where(fc._is_triplets_included==False)[0]])
        fc.get_irreducible_fc3_components_with_spg()
        # for tp in range(len(fc3_ir_triplets)):
        #     print (np.dot(fc._ifc3_trans[tp], fc3_ir_triplets.flatten()[fc._ifc3_ele]) - fc3_ir_triplets[tp].flatten()).round(6)
        print "spg invariance reduces 3rd IFC from %d to %d" %(np.sum(fc._is_triplets_included)*27, len(fc._ifc3_ele))



        # np.set_printoptions(formatter={'float':lambda x: format(x, '7.4f')})
        # for triplet in [(4,4,4), (4, 5, 10), (38, 24, 10), (22, 27, 39)]:
        #     print np.dot(fc._coeff3[triplet], np.dot(fc._ifc3_trans[fc._ifc3_map[triplet]],
        #                                              fc3_ir_triplets.flatten()[fc._ifc3_ele])).round(6) -\
        #           fc3[triplet].flatten().round(6)

        if self._is_trans_inv:
            fc.get_fc3_translational_invariance()
            print "translational invariance reduces 3rd IFC to %d"%(len(fc._ifc3_ele))
        if self._is_rot_inv and self.fc2 is not None:
            fc.get_fc3_rotational_invariance(self._fc2)
            print "rotational invariance reduces 3rd IFC to %d"%(len(fc._ifc3_ele))
        self._num_irred_fc3 = len(fc._ifc3_ele)
        sys.stdout.flush()

    @timeit
    def calculate_irreducible_fc3(self):
        fc = self._fc
        coeff = fc._coeff3
        ifcmap = fc._ifc3_map
        trans = fc._ifc3_trans
        num_irred = trans.shape[-1]
        num_step = len(self._displacements)
        included = fc._is_triplets_included
        # #
        # import h5py
        # f=h5py.File("fc3-direct.hdf5")
        # fc3=f['fc3'].value
        # f.close()
        # fc3_pair = np.array([fc3[i] for i in fc._triplets_reduced])
        # fc3_reduced = fc3_pair.flatten()[fc._ifc3_ele]
        # f=h5py.File("fc2.hdf5")
        # fc2=f['fc2'].value
        # f.close()
        # fc2_pair = np.array([fc2[i] for i in fc._pairs_reduced])
        # fc2_reduced = fc2_pair.flatten()[fc._ifc2_ele]
        print "Calculating fc3..."
        sys.stdout.flush()
        import _mdfc as mdfc
        # ddcs3 = np.zeros((num_step * self._num_atom * 3, num_irred), dtype="double")
        ddcs = np.zeros((num_step, self._num_atom, 3, num_irred), dtype="double") #displacements  rearrangement as the coefficients of fc3
        mdfc.rearrange_disp_fc3(ddcs,
                                self._displacements,
                                coeff,
                                included.astype("int8"),
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
        # self._forces2 = self._forces1 - 1. / 2. * np.dot(ddcs, fc3_reduced)
        # self._fc3_irred = fc3_reduced
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




    def set_cutoffs(self, cutoff_pair=None, cutoff_triplet=None):
        specie, sequence = np.unique(self._unitcell.get_atomic_numbers(), return_index=True)
        self._cutoff = Cutoff(specie[np.argsort(sequence)], cutoff_pair, cutoff_triplet)
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

    def _set_symmetry(self):
        self._symmetry = Symmetry(self._supercell,
                                  self._symprec,
                                  self._is_symmetry)



    def set_force_constants(self, force_constants):
        self._fc2 = force_constants

    def set_force_sets(self, sets_of_forces_objects):
        self._set_of_forces_objects = sets_of_forces_objects

    def get_fc2(self):
        return self._fc2
    fc2 = property(get_fc2)



class Cutoff():
    def __init__(self, species, cut_pair, cut_triplets):
        self._cell = None
        self._pair_distances = None
        n = len(species)

        # cutoff pair
        cp = np.ones((n, n), dtype="float") * 10000
        if cut_pair is not None:
            if len(cut_pair) == (n +1) * n / 2:
                for i in range(n):
                    for j in range(i,n):
                        cp[i,j] = cp[j,i] = cut_pair.pop(0)
                self._cut_pair = cp
            elif len(cut_pair) == n:
                for i, j in np.ndindex((n,n)):
                    cp[i,j] = np.average([cut_pair[i], cut_pair[j]])
            elif len(cut_pair) == 1:
                for i, j in np.ndindex((n,n)):
                    cp[i,j] = cut_pair[0]
            else:
                print_error_message("Cutoff pairs not equal to the number needed [1, n or (n+1)*n/2]!")
        else:
            self._cut_pair = None

        # cutoff triplet
        if cut_triplets is not None:
            ct = np.ones((n,n,n), dtype="float") * 10000
            if len(cut_triplets) == n * (n + 1) * (n + 2) / 6:
                for i in range(n):
                    for j in range(i,n):
                        for k in range(j,n):
                            ct[i,j,k] = ct[i,k,j] = ct[j,i,k] = ct[j,k,i] =\
                                ct[k,i,j] = ct[k,j,i] = cut_triplets.pop(0)
            elif len(cut_triplets) == (n +1) * n / 2:
                cp = np.ones((n, n), dtype="float") * 10000
                for i in range(n):
                    for j in range(i,n):
                        cp[i,j] = cp[j,i] = cut_triplets.pop(0)
                for i,j,k in np.ndindex((n,n,n)):
                    ct[i,j,k] = np.average([cp[i,j], cp[i,k], cp[j,k]])
            elif len(cut_triplets) == n:
                for i,j,k in np.ndindex((n,n,n)):
                    ct[i,j,k] = np.average([cut_triplets[i], cut_triplets[j], cut_triplets[k]])
            elif len(cut_triplets) == 1:
                for i,j,k in np.ndindex((n,n,n)):
                    ct[i,j,k] = cut_triplets[0]
            else:
                print_error_message("Cutoff triplets not equal to the number needed[1, n, (n+1)*n/2 or n*(n+1)*(n+2)/6]!")
            self._cut_triplet = ct
        else:
            self._cut_triplet = None

    def set_cell(self, cell, symprec = 1e-5):
        self._cell = cell
        self._symprec = symprec
        self._pair_distances = None

    def get_cutoff_pair(self):
        return self._cut_pair


    def get_cutoff_triplet(self):
        return self._cut_triplet

    def expand_pair(self):
        unique_atoms, index_unique = np.unique(self._cell.get_atomic_numbers(), return_index=True)
        unique_atoms = unique_atoms[np.argsort(index_unique)] # in order to keep the specie sequence unchanged
        if self.get_cutoff_pair() is not None:
            cutpair_expand = np.zeros((self._cell.get_number_of_atoms(), self._cell.get_number_of_atoms()), dtype="double")
            for i in range(self._cell.get_number_of_atoms()):
                index_specie_i = np.where(unique_atoms == self._cell.get_atomic_numbers()[i])[0]
                for j in range(i, self._cell.get_number_of_atoms()):
                    index_specie_j = np.where(unique_atoms == self._cell.get_atomic_numbers()[j])[0]
                    cutpair_expand[i,j] = cutpair_expand[j,i] = self._cut_pair[index_specie_i, index_specie_j]
        else:
            cutpair_expand = None
        return cutpair_expand

    def expand_triplet(self, triplets=None):
        natom = self._cell.get_number_of_atoms()
        unique_atoms, index_unique = np.unique(self._cell.get_atomic_numbers(), return_index=True)
        unique_atoms = unique_atoms[np.argsort(index_unique)] # in order to keep the specie sequence unchanged
        if self.get_cutoff_triplet() is not None:
            if triplets is not None:
                cut_triplet_expand = np.zeros(len(triplets), dtype="double")
                for t, triplet in enumerate(triplets):
                    i, j, k = triplet
                    index_specie_i = np.where(unique_atoms == self._cell.get_atomic_numbers()[i])[0]
                    index_specie_j = np.where(unique_atoms == self._cell.get_atomic_numbers()[j])[0]
                    index_specie_k = np.where(unique_atoms == self._cell.get_atomic_numbers()[k])[0]
                    cut_temp  = self._cut_triplet[index_specie_i, index_specie_j, index_specie_k]
                    cut_triplet_expand[t] = cut_temp
                    # cut_triplet_expand[i,j, k] =  self._cut_triplet[index_specie_i, index_specie_j, index_specie_k]
            else:
                cut_triplet_expand = np.zeros((natom, natom, natom), dtype="double")
                for i in range(natom):
                    index_specie_i = np.where(unique_atoms == self._cell.get_atomic_numbers()[i])[0]
                    for j in range(i, natom):
                        index_specie_j = np.where(unique_atoms == self._cell.get_atomic_numbers()[j])[0]
                        for k in range(j, natom):
                            index_specie_k = np.where(unique_atoms == self._cell.get_atomic_numbers()[k])[0]
                            cut_temp  = self._cut_triplet[index_specie_i, index_specie_j, index_specie_k]
                            cut_triplet_expand[i,j,k] = cut_temp
                            cut_triplet_expand[j,i,k] = cut_temp
                            cut_triplet_expand[i,k,j] = cut_temp
                            cut_triplet_expand[j,k,i] = cut_temp
                            cut_triplet_expand[k,i,j] = cut_temp
                            cut_triplet_expand[k,j,i] = cut_temp
                            # cut_triplet_expand[i,j, k] =  self._cut_triplet[index_specie_i, index_specie_j, index_specie_k]
        else:
            cut_triplet_expand = None
        return cut_triplet_expand

    def set_pair_distances(self, triplets=None):
        num_atom = self._cell.get_number_of_atoms()
        lattice = self._cell.get_cell()
        min_distances = np.zeros((num_atom, num_atom), dtype='double')
        if triplets is not None:
            for triplet in triplets:
                for permute in [(0,1), (1,2), (0,2)]:
                    i, j = triplet[permute[0]], triplet[permute[1]]
                    min_distances[i, j] = min_distances[j, i] =\
                        np.linalg.norm(np.dot(
                            get_equivalent_smallest_vectors(
                                i, j, self._cell, lattice, self._symprec)[0], lattice))
        else:
            for i in range(num_atom): # run in cell
                for j in range(num_atom): # run in primitive
                    min_distances[i, j] = np.linalg.norm(np.dot(
                            get_equivalent_smallest_vectors(
                                i, j, self._cell, lattice, self._symprec)[0], lattice))
        self._pair_distances = min_distances

    def get_pair_inclusion(self):
        num_atom = self._cell.get_number_of_atoms()
        cut_pair = self.expand_pair()
        include_pair= np.ones((num_atom, num_atom), dtype=bool)
        if self._pair_distances == None:
            self.set_pair_distances()
        for i, j in np.ndindex(num_atom, num_atom):
            if cut_pair is not None:
                ave_dist = (self._pair_distances[i,j] + self._pair_distances[j,i]) / 2.
                if ave_dist > cut_pair[i,j]:
                    include_pair[i,j] = False
        return include_pair

    def get_coefficient_weight(self, pair_or_triplets, weight=None):
        if self._pair_distances is None:
            self.set_pair_distances()
        if weight is None:
            weight = 1.
        weights = np.zeros(len(pair_or_triplets), dtype="double")
        for i, pt in enumerate(pair_or_triplets):
            if len(pt) == 2:
                a, b = pt
                dist  = self._pair_distances[a, b]
            if len(pt) == 3:
                a, b, c  = pt
                dist = np.average(self._pair_distances[a, b], self._pair_distances[b, c], self._pair_distances[a, c])
            weights[i] = np.exp(dist) * weight
        return weights




    def get_triplet_inclusion(self, triplets):
        num_atom = self._cell.get_number_of_atoms()
        cut_triplet = self.expand_triplet(triplets)

        if self._pair_distances == None:
            self.set_pair_distances()
        include_triplet= np.ones(len(triplets), dtype=bool)
        for t, triplet in enumerate(triplets):
            i, j, k  = triplet
            if cut_triplet is not None:
                a, b, c = self._pair_distances[i,j], self._pair_distances[j,k], self._pair_distances[i,k]
                dist = (a + b + c) / 3.
                # max_dist = max(self._pair_distances[i,j], self._pair_distances[j,k],self._pair_distances[i,k])
                if dist > cut_triplet[t]:
                    include_triplet[t] = False
        return include_triplet