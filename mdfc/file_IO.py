from realmd.mddos import MolecularDynamicsCoordinateVelocity

__author__ = 'xinjiang'
import os
import numpy as np
from realmd.information import warning, error
from realmd.unit import au_energy, au_velocity

def read_fc2_from_hdf5(filename):
    import h5py
    f = h5py.File(filename, 'r')
    if 'fc2' in f.keys():
        fc2 = f['fc2'][:]
    else:
        fc2 = f['force_constants'][:]
    f.close()
    return fc2

def read_fc3_from_hdf5(filename):
    import h5py
    f = h5py.File(filename, 'r')
    fc3 = f['fc3'][:]
    f.close()
    return fc3

def get_volume_from_md_file(filename="md.out"):
    if not os.path.exists(filename):
        print "file %s does not exist!"%filename
        return
    with open(filename, "r") as f:
        md=f.readlines()
    volume_lines = filter(lambda line: line.find("Volume:")!=-1, md)
    volume = np.array([line.strip().split()[1] for line in volume_lines], dtype="double")
    return volume

def read_heat_flux(format="x", filename=None, step_range = slice(None), is_out_step=True):
    if format =="x":
        if filename == None:
            filename="heat_flux.out"
        return read_heat_flux_xyz(filename=filename, step_range=step_range, is_out_step=is_out_step)
    elif format == "l":
        if filename == None:
            filename = "heatflux.dat"
        return read_heat_flux_lammps(filename=filename, step_range=step_range, is_out_step=is_out_step)


def read_heat_flux_xyz(filename="heat_flux.out", step_range = slice(None)):
    if not os.path.exists(filename):
        error("file %s does not exist!"%filename)
    with open(filename, "r") as f:
        md=f.read()
    if md.find("Heat Flux") == -1:
        error("file %s does not contain information about heat flux")
    hf = np.array([np.fromstring(line.strip().split("\n")[0],dtype="double", sep=" ")
                   for line in md.split("Heat Flux (au)")[1:]])

    return hf[step_range] * au_energy * au_velocity

def read_heat_flux_lammps(filename="heatflux.dat", step_range = slice(None), is_out_step=True):
    if not os.path.exists(filename):
        error("file %s does not exist!"%filename)
    with open(filename, "r") as f:
        md=f.readlines()
    if md[0].find("Jx") == -1 or md[0].find("Jy") == -1 or md[0].find("Jz")==-1:
        error("file %s does not contain information about heat flux")
    index = [md[0].split().index(l) for l in ("Jx","Jy","Jz")]
    data = np.array([num.split() for num in md[1:]], dtype="double")
    hf = data[:,index]
    if is_out_step:
        step = data[:,0]
        return hf[step_range], step[step_range]
    else:
        return hf[step_range]

def write_fc3_hdf5(fc3, filename="fc3.hdf5"):
    import h5py
    f = h5py.File(filename, "w")
    f.create_dataset("fc3", data=fc3)
    f.close()

def write_fc2_hdf5(fc2, filename="fc2.hdf5"):
    import h5py
    f = h5py.File(filename, "w")
    f.create_dataset("fc2", data=fc2)
    f.close()

class MD_fec(MolecularDynamicsCoordinateVelocity):
    "Force, energy and coordinate information read from MD steps"
    def __init__(self, step_range=slice(None)):
        MolecularDynamicsCoordinateVelocity.__init__(self, step_range=step_range)
        self.atom_forces = None #unit in au (Hartree/Bohr
        self.atom_energies = None #unit in Hartree
        self.volume = None

    def _set_from_file(self, fileformat="x", cv_filename="geo_end.xyz", fe_filename="fe.out"):
        if cv_filename == None:
            cv_filename = "geo_end.xyz"
        if fe_filename == None:
            fe_filename = "fe.out"
        MolecularDynamicsCoordinateVelocity.set_from_file(self,fileformat=fileformat, filename=cv_filename)
        if not os.path.exists(fe_filename):
            print "Reading force and energy information"
            print "file:%s does not exist"%fe_filename
            return 0
        return self.read_fe_file(filename=fe_filename)

    def read_fe_file(self, filename):
        if self.fileformat == "x":
            self.read_xyz_fe_file(filename)
        elif self.fileformat == "l":
            self.read_lammps_fe_file(filename)

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
        "unit of force: Ha/Bohr (au), energy: Hartree (au)"
        input_file=file(filename,'r')
        file_all=input_file.read()
        file_list_all=file_all.split('MD step:')[1:]
        assert len(file_list_all) == self.total_num_steps
        file_list=file_list_all[self.step_range]
        self.atom_energies = np.zeros((self.num_steps, self.num_atom))
        self.atom_stresses = np.zeros((self.num_steps, self.num_atom, 3, 3), dtype="double")
        self.atom_forces = np.zeros((self.num_steps, self.num_atom,3))

        for i, step in enumerate(file_list):
            stress_lines = step.split("Atom resolved stress tensors")[-1].strip().split("\n")[:self.num_atom]
            stress = np.array([np.fromstring(s, sep=" ", dtype="double") for s in stress_lines])
            self.atom_stresses[i] = stress.reshape(self.num_atom, 3,3)
            if step.find("Atom resolved total energies") == -1:
                return 0
            energies = step[step.rfind("Atom resolved total energies"):].split("\n")[1:self.num_atom+1]
            self.atom_energies[i] = np.array([line.split()[1] for line in energies], dtype="double") * au_energy
        input_file.close()
        return 1

    def _save2hdf5(self, filename="md_fecv.hdf5"):
        "save the force, energy, coordinate and velocity into a hdf5 file"
        try:
            import h5py
        except ImportError:
            warning("h5py not implemented, velocities not saved")
            return
        o=h5py.File(filename,"w")
        if self.volume is not None:
            o.create_dataset("volume", data=self.volume)
        o.create_dataset("velocities", data=self.atom_velocities)
        o.create_dataset("coordinates", data=self.atom_coordinates)
        o.create_dataset("forces", data=self.atom_forces)
        o.create_dataset("energies",data=self.atom_energies)
        o.create_dataset("atom_types",data=self.atom_types)
        o.create_dataset("step_indices",data=self.step_indices)
        o.close()
