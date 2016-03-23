import numpy as np
import os
from realmd.file_IO import read_xyz_fe_file, write_md_to_hdf5, read_md_from_hdf5, read_md_from_vasprun
from realmd.mddos import MolecularDynamicsCoordinateVelocity as MDCV
from realmd.unit import au_energy, au_force
from realmd.information import warning


class MolecularDynamicsForceEnergy(MDCV):
    "Force, energy and coordinate information read from MD steps"
    def __init__(self, step_range=slice(None), time_step=None):
        MDCV.__init__(self, step_range=step_range, time_step=time_step)
        self.atom_forces = None #unit in au (Hartree/Bohr
        self.atom_energies = None #unit in Hartree
        self.atom_stresses = None
        self.volume = None

    def _set_from_file(self, fileformat="x", cv_filename="geo_end.xyz", fe_filename="fe.out"):
        "cv_filename: coordinates and velocities filename; fe_filename: forces and energies filename"
        if cv_filename == None:
            cv_filename = "geo_end.xyz"
        if fe_filename == None:
            fe_filename = "fe.out"
        MDCV.set_from_file(self,fileformat=fileformat, filename=cv_filename)
        self.print_information()
        if not os.path.exists(fe_filename):
            print "Reading force and energy information"
            print "file:%s does not exist"%fe_filename
            return 0
        return self.read_fe_file(filename=fe_filename)

    def read_fe_file(self, filename):
        parameters = {}
        if self.fileformat == "x":
            parameters = read_xyz_fe_file(filename, self.num_atom, step_range=self.step_range)
        elif self.fileformat == "l":
            self.read_lammps_fe_file(filename)
        elif self.fileformat == "h":
            parameters = read_md_from_hdf5(filename)
        elif self.fileformat == "v":
            parameters = read_md_from_vasprun(filename)
        if self.fileformat == "h":
            _slice = self.step_range
        else:
            _slice = slice(None)
        # it is supposed that those units has already been converted in the subroutine
        if "energies" in parameters.keys():
            self.atom_energies = parameters['energies'][_slice]
        if "forces" in parameters.keys():
            self.atom_forces = parameters['forces'][_slice]
        if "stresses" in parameters.keys():
            self.atom_stresses = parameters['stresses'][_slice]
        return 1

    def read_lammps_fe_file(self, filename):
        input_file=file(filename,'r')
        file_all=input_file.read()
        file_list=file_all.split('ITEM: TIMESTEP\n')[1:]
        file_list=file_list[self.step_range]
        self.atom_energies = np.zeros((self.num_steps, self.num_atom), dtype="double")
        self.atom_stresses = np.zeros((self.num_steps, self.num_atom, 3, 3), dtype="double")
        for i, step in enumerate(file_list):
            self.step_indices[i] = int(step.strip().split('\n')[0])
            left=step.split('ITEM: ATOMS')[-1].strip().split("\n")
            out_inf = left[0].strip().split()
            bulk_inf=np.array([a.split() for a in left[1:]], dtype="double")
            if "v_TOTE" in out_inf:
                e_index = out_inf.index("v_TOTE")
                self.atom_energies[i] = bulk_inf[:,e_index]
            else:
                ke_index = out_inf.index("c_myKE")
                pe_index = out_inf.index("c_myPE")
                potential_energy = bulk_inf[:,pe_index]
                kinetic_energy = bulk_inf[:,ke_index]
                self.atom_energies[i] = potential_energy + kinetic_energy
            if "v_Sxx" in out_inf:
                xx_index = out_inf.index("v_Sxx")
                yy_index = out_inf.index("v_Syy")
                zz_index = out_inf.index("v_Szz")
                xy_index = out_inf.index("v_Sxy")
                xz_index = out_inf.index("v_Sxz")
                yz_index = out_inf.index("v_Syz")
                self.atom_stresses[i,:,0,0] = bulk_inf[:,xx_index]
                self.atom_stresses[i,:,1,1] = bulk_inf[:,yy_index]
                self.atom_stresses[i,:,2,2] = bulk_inf[:,zz_index]
                self.atom_stresses[i,:,0,1] = bulk_inf[:,xy_index]
                self.atom_stresses[i,:,1,0] = bulk_inf[:,xy_index]
                self.atom_stresses[i,:,0,2] = bulk_inf[:,xz_index]
                self.atom_stresses[i,:,2,0] = bulk_inf[:,xz_index]
                self.atom_stresses[i,:,2,1] = bulk_inf[:,yz_index]
                self.atom_stresses[i,:,1,2] = bulk_inf[:,yz_index]
        input_file.close()

    def read_xyz_fe_file(self,filename):
        parameters = read_xyz_fe_file(filename, self.num_atom, step_range=self.step_range)
        if "energy" in parameters.keys():
            self.atom_energies = parameters['energy'] * au_energy
        if "force" in parameters.keys():
            self.atom_forces = parameters['force'] * au_force
        if "stress" in parameters.keys():
            self.atom_stresses = parameters['stress']
        return 1

    def save_fe_to_hdf5(self, filename="md_fe.hdf5"):
        "save the force, energy, coordinate and velocity into a hdf5 file"
        try:
            import h5py
            write_md_to_hdf5(filename=filename,
                             force=self.atom_forces,
                             energy=self.atom_energies,
                             stress=self.atom_stresses)
        except ImportError:
            warning("h5py not implemented, velocities not saved")

