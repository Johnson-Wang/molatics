import numpy as np
import os
from realmd.file_IO import find_pos, read_xyz_cv_file, read_md_from_hdf5, write_md_to_hdf5, get_positions_vasprun_xml,\
    get_atom_types_from_vasprun_xml
from realmd.information import warning, error


class MolecularDynamicsCoordinateVelocity():
    def __init__(self, step_range=slice(None), time_step=None):
        self.num_steps=None
        self.num_atom=None
        self.step_indices=None
        self.atom_ids=None
        self.atom_types=None
        self.total_num_steps=None
        self.step_range=step_range
        self.atom_coordinates = None
        self.atom_coordinates_pre = self.atom_coordinates_post = None
        self.atom_velocities = None
        self.fileformat = 'x'
        self.lattice_bounds = None
        self.time_step=None

    def print_information(self):
        print
        print "#"*75
        print "##information in the reading process"
        if self.total_num_steps is not None:
            print "total time steps in the input file:%d" %self.total_num_steps
        if self.step_range != slice(None):
            ran=self.step_range
            start = 0 if ran.start is None else ran.start
            stop = -1 if ran.stop is None else ran.stop
            step = 1 if ran.step is None else ran.step
            print "extracted time steps: from %d to %d with gap %d" % (start, stop, step)
        else:
            print "extracted time steps: from %d to the end with gap %d" %(0, 1)
        print "Total number of the extracted steps is %d" %self.num_steps
        # if self.step_indices != None:
        #     print "time interval between two steps:%d" %((self.step_indices[1]-self.step_indices[0])*gap)
        print "#"*75
        print

    def set_from_file(self, fileformat="l", filename=""):
        self.fileformat = fileformat
        print "Reading velocity and coordinate information"
        if not os.path.exists(filename):
            print "file:%s does not exist"%filename
            return 0
        else:
            print "from file: %s" %filename
        if fileformat=="l":
            return self.read_lammps_file(filename)
        elif fileformat=="x":
            return self.read_xyz_file(filename)
        elif fileformat=="v":
            return self.read_vasp_file(filename)
        elif fileformat=="h":
            return self.read_hdf5_file(filename)


    def read_lammps_file(self, filename):
        input_file=file(filename,'r')
        file_all=input_file.read()
        if file_all[:15] != ("ITEM: TIMESTEP\n"):
            print "the format of input file %s does not obey the lammps format"
            return 0
        file_list=file_all.split('ITEM: TIMESTEP\n')[1:]
        self.total_num_steps=len(file_list)
        file_list=file_list[self.step_range]
        self.num_steps=len(file_list)
        # num_atoms=len(file_list[0].strip().split("ITEM: ATOMS id vx vy vz \n")[-1].split('\n'))
        file_list_0 = file_list[0].split("\n")
        num_atoms = int(file_list_0[find_pos("ITEM: NUMBER OF ATOMS", file_list_0)+1])
        self.num_atom=num_atoms
        self.step_indices = np.zeros(self.num_steps,dtype=int)
        self.atom_velocities = np.zeros((self.num_steps, num_atoms,3), dtype="double")
        self.atom_coordinates = np.zeros_like(self.atom_velocities)
        for i, step in enumerate(file_list):
            self.step_indices[i] = int(step.strip().split('\n')[0])
            left=step.split('ITEM: ATOMS')[-1].strip().split("\n")
            out_inf = left[0].strip().split()
            v_index = [out_inf.index(a) for a in ("vx", "vy", "vz")]
            c_index = [out_inf.index(a) for a in ("x", "y", "z")]
            bulk_inf=np.array([a.split() for a in left[1:]], dtype="double")
            assert bulk_inf.shape[0] == self.num_atom and bulk_inf.shape[1] == len(out_inf)
            velocities = bulk_inf[:,v_index]
            coordinates = bulk_inf[:,c_index]
            self.atom_velocities[i] = velocities
            self.atom_coordinates[i] = coordinates
        lattice_bounds = step.split("ITEM: BOX BOUNDS")[-1].strip().split("\n")[1:4]
        self.lattice_bounds = np.array([l.split() for l in lattice_bounds], dtype=float)
        self.atom_ids=list(bulk_inf[:,out_inf.index("id")].astype(dtype="int"))
        self.atom_types=np.zeros(len(self.atom_ids), dtype=int)
        irred_atoms=[]
        for i, a in enumerate(self.atom_ids):
            if a not in irred_atoms:
                self.atom_types[i]=len(irred_atoms)
                irred_atoms.append(a)
            else:
                self.atom_types[i]=irred_atoms.index(a)
        input_file.close()
        return 1


    def read_xyz_file(self, filename):
        "the unit of velocity is Angstrom/ps, coordinates: Angstrom"
        parameters = read_xyz_cv_file(filename=filename, step_range=self.step_range)
        self.atom_velocities = parameters['velocity']
        self.atom_coordinates = parameters['coordinate']
        self.atom_types = parameters['atom_type']
        self.step_indices = parameters['step_indices']
        self.num_steps = parameters['num_step']
        self.total_num_steps = len(self.atom_coordinates)
        self.num_atom = len(self.atom_types)

    def read_vasp_file(self, filename):
        if filename==None:
            filename="vasprun.xml"
        if filename == "XDATCAR":
            xdat=open(filename, "r")
            all_data=xdat.read().split("Direct configuration=")
            structure=all_data[0].split("\n")
            lc = float(structure[1]) # lattice constant
            lattice = np.array([np.fromstring(s, dtype="double", sep=" ") for s in structure[2:5]])
            lattice*=lc
            species=np.array(structure[5].split())
            specie_num=np.array(map(int,structure[6].split()))
            self.atom_ids=sum(map(lambda x,y:[x]*y,species, specie_num), [])
            self.num_atom=len(self.atom_ids)
            self.atom_types=self.atom_ids
            mdata=[mdrun.split("\n") for mdrun in all_data[1:]]
            step_indices=np.array([m[0] for m in mdata[:-1]],dtype="int")
            step=(step_indices[1]-step_indices[0])
            pos=np.array([[p.split() for p in ps[1:-1]] for ps in mdata],dtype="double")
            for i, j in np.ndindex(pos.shape[:2]):
                pos[i,j]=np.dot(lattice, pos[i,j])
            velocities=(pos[1:]-pos[:-1])/step # unit of AA/fs
            self.atom_velocities=velocities[self.step_range]
            self.step_indices=step_indices[:-1][self.step_range]
            self.total_num_steps=len(step_indices)-1
            self.num_steps=len(self.step_indices)
        else:
            positions = get_positions_vasprun_xml(filename, step_range=self.step_range)
            self.atom_coordinates = positions
            self.num_steps=len(self.atom_coordinates)
            atom_types, masses, num_atom = get_atom_types_from_vasprun_xml(filename)
            self.atom_types = atom_types
            self.num_atom = num_atom


    def read_hdf5_file(self,filename):
        try:
            import h5py
            parameters = read_md_from_hdf5(filename)
            if "coordinates" in parameters.keys():
                self.atom_coordinates = parameters['coordinates'][self.step_range]
            if "atom_types" in parameters.keys():
                self.atom_types = parameters["atom_types"]
            if "time_step" in parameters.keys():
                self.time_step = parameters['time_step']
            if "velocities" in parameters.keys():
                self.velocities = parameters["velocities"][self.step_range]
            self.num_steps,self.num_atom=self.atom_coordinates.shape[0:2]
            self.atom_ids=np.zeros(self.num_atom,dtype=int)
            return 1
        except ImportError:
            print "h5py not implemented"
            return 0
        except IndexError:
            print "The step range chosen is out of range!"
            return 0

    def save_cv_to_hdf5(self, filename="md_cv.hdf5"):
        "Save the coordinate and velocity information into a hdf5 file"
        try:
            import h5py
            write_md_to_hdf5(filename=filename,
                             coordinate=self.atom_coordinates,
                             velocity=self.atom_velocities,
                             type=self.atom_types,
                             time_step=self.time_step)
            print "Coordinates and velocity information has been written to %s" %filename
        except ImportError:
            warning("h5py not implemented, velocities not saved")
