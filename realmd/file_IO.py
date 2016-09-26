# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:58:13 2013

@author: xwangan
"""
import os
import numpy as np
import time
import subprocess
import copy
from itertools import islice
from realmd.information import error
from realmd.unit import au_energy, au_velocity, au_force
import StringIO

class VasprunWrapper(object):
    """VasprunWrapper class
    This is used to avoid VASP 5.2.8 vasprun.xml defect at PRECFOCK,
    xml parser stops with error.
    """
    def __init__(self, f):
        self.f = f

    def read(self, size=None):
        element = self.f.next()
        if element.find("PRECFOCK") == -1:
            return element
        else:
            return "<i type=\"string\" name=\"PRECFOCK\"></i>"

def find_pos(str, str_list):
    for i,line in enumerate(str_list):
        if line.find(str) != -1:
            return i
    return None

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def pass_lines(file_handle, line_num, return_lines=False):
    if not return_lines:
        for i in range(line_num):
            file_handle.next()
        return line_num
    else:
        lines =[]
        for i in range(line_num):
            lines.append(file_handle.next())
        return line_num, lines

def pass_lines2(file_handle, line_num, return_lines=False):
    if not return_lines:
        lines = list(islice(file_handle, line_num))
        return line_num
    else:
        lines = list(islice(file_handle, line_num))
        return line_num, lines

def get_forces_vasprun_xml(vasprun):
    from lxml import etree
    f = open(vasprun)
    vasprun = etree.iterparse(f, tag='varray', events=("end",))
    forces = []
    for event, element in vasprun:
        if element.attrib['name'] == 'forces':
            force = []
            for v in element.xpath('./v'):
                force.append([float(x) for x in v.text.split()])
            forces.append(force)
    f.close()

    return np.array(forces, dtype="double")

def get_positions_vasprun_xml(vasp_filename, step_range=slice(None)):
    start = 0 if step_range.start is None else step_range.start
    stop = -1 if step_range.stop is None else step_range.stop
    step = 1 if step_range.step is None else step_range.step
    from lxml import etree
    f = open(vasp_filename, "r")
    lines = f.read()
    if lines.strip()[-100:].find("</modeling>") == -1:
        if lines.strip()[-8:] == "<scstep>":
            lines += "  </scstep>\n"
        lines += "  </calculation>\n"
        lines += "</modeling>\n"
    f.close()
    f = StringIO.StringIO(lines)
    vasprun = etree.iterparse(f, tag='structure')
    positions = []
    n = 0
    for event, element in vasprun:
        if element.get('name') is not None:
            continue

        n += 1
        if n < start or (n-start) % step != 0:
            continue

        position = []
        for ele in element.findall('varray'):
            if ele.get("name") == 'positions':
                for v in ele.xpath('./v'):
                    position.append([float(x) for x in v.text.split()])
                positions.append(position)
        if n == stop:
            break


    return np.array(positions, dtype="double")

def get_atom_types_from_vasprun_xml(vasp_filename):
    from lxml import etree
    f = open(vasp_filename, "r")
    vasprun = etree.iterparse(VasprunWrapper(f), tag='array')
    atom_types = []
    masses = []
    num_atom = 0
    for ele, element in vasprun:
        if 'name' in element.attrib:
            if element.attrib['name'] == 'atomtypes':
                for rc in element.xpath('./set/rc'):
                    atom_info = [x.text for x in rc.xpath('./c')]
                    num_atom += int(atom_info[0])
                    atom_types.append(atom_info[1].strip())
                    masses += ([float(atom_info[2])] * int(atom_info[0]))
                break
    f.close()
    return atom_types, masses, num_atom

def check_file_length(total_lines, one_step_lines, start=0, stop=None, step=1):
    total_steps = total_lines / one_step_lines
    if start >= total_steps:
        return False
    elif stop == None:
        return True
    elif (stop - start) / step > total_steps:
        return False
    return True

def read_xyz_cv_file(filename, step_range=slice(None)):
    "the unit of velocity is Angstrom/ps, coordinates: Angstrom"
    x=file(filename,'r')
    try:
        num_atoms=int(x.readline())
    except:
        print "the format of input file %s does not obey the xyz format"
        return 0
    parameters = {}
    parameters.setdefault("num_atom", num_atoms)
    xall = x.read()
    all_steps = xall.split('MD iter:')[1:]
    try:
        sliced_steps = all_steps[step_range]
    except IndexError:
        error("Invalid range in reading %s!" %filename)
    num_steps=len(sliced_steps)
    atom_velocities=np.zeros((num_steps, num_atoms, 3))
    atom_coordinates = np.zeros_like(atom_velocities)
    step_indices = np.zeros(num_steps, dtype="intc")
    for index,one_step in enumerate(sliced_steps):
        one_step_list = one_step.split("\n")[:num_atoms+1]
        step_indices[index] = int(one_step_list[0])
        coordinates = [map(float, v.split()[1:4]) for v in one_step_list[1:]]
        atom_coordinates[index] = np.array(coordinates)
        velocities=[map(float, v.split()[-3:]) for v in one_step_list[1:]]
        atom_velocities[index]=np.array(velocities)
    atom_types=[id.strip().split()[0] for id in one_step_list[1:]]
    parameters.setdefault('num_step', len(all_steps))
    parameters.setdefault("velocity", atom_coordinates)
    parameters.setdefault('coordinate', atom_coordinates)
    parameters.setdefault('atom_type', atom_types)
    parameters.setdefault('step_indices', step_indices)
    return parameters

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


def read_heat_flux_xyz(filename="heat_flux.out", step_range = slice(None), is_out_step=True):
    if not os.path.exists(filename):
        error("file %s does not exist!"%filename)
    with open(filename, "r") as f:
        md=f.read()
    if md.find("Heat Flux") == -1:
        error("file %s does not contain information about heat flux")
    hf = np.array([np.fromstring(line.strip().split("\n")[0],dtype="double", sep=" ")
                   for line in md.split("Heat Flux (au)")[1:]])
                   # for line in md.split("HF_ve (au)")[1:]])
                   # for line in md.split("HF_electric (au)")[1:]])
                   # for line in md.split("HF_dispersion (au)")[1:]])
                   # for line in md.split("HF_repulsive (au)")[1:]])
                   # for line in md.split("HF_virial (au)")[1:]])
    if is_out_step:
        step = np.array([int(line.strip().split("\n")[0])
                         for line in md.split("MD step:")[1:]])

        return hf[step_range] * au_energy * au_velocity, step[step_range]
    else:
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

def get_one_step_number_of_lines(filename):
    with open(filename) as input_handle:
        mode = []
        num_mode_line = []
        input_handle.seek(0)
        first_step_covered = False
        num_line_second_step = -1
        for i, line in enumerate(input_handle.xreadlines()):
            # line = input_file.readline()
            if line.find("MD step") != -1:
                if first_step_covered:
                    num_line_second_step = i
                    break
                mode.append("step")
                num_mode_line.append(i)
                first_step_covered = True
            if line.find("Atom resolved stress tensors")!=-1:
                mode.append("tensor")
                print "Tensor information contained in the input file %s" %input_handle.name
                num_mode_line.append(i)
            if line.find("Atom resolved total energies") != -1:
                mode.append("energy")
                print "Energy information contained in the input file %s" %input_handle.name
                num_mode_line.append(i)
            if line.find("Total Forces") != -1:
                mode.append("force")
                print "Force information contained in the input file %s" %input_handle.name
                num_mode_line.append(i)
            if line.find("Heat Flux") != -1:
                mode.append("heatflux")
                print "Heat flux information contained in the input file %s" %input_handle.name
                num_mode_line.append(i)
        sorted_couple = sorted(zip(num_mode_line, mode))
        num_mode_line = [m for (m, n) in sorted_couple]
        mode = [n for (m, n) in sorted_couple]
        # the extra element in the num_mode_line is useful because it represents the start of the next step
    return  mode, num_mode_line + [num_line_second_step]

# @timeit
# def read_xyz_fe_file(filename, num_atom, return_info=None, step_range=slice(None)):
#     "unit of force: Ha/Bohr (au), energy: Hartree (au)"
#     if return_info == None:
#         return_info = ['force', 'energy', 'stress', 'heatflux']
#     total_lines = file_len(filename)
#     mode, num_mode_line = get_one_step_number_of_lines(filename)
#     num_line_one_step = num_mode_line[-1] - num_mode_line[0]
#     start, stop, step = (step_range.start, step_range.stop, step_range.step)
#     start = start if start is not None else 0
#     step = step if step is not None else 1
#     stop = stop if stop is not None else total_lines / num_line_one_step
#     if not check_file_length(total_lines, num_line_one_step, start, stop, step):
#         return None
#
#     data_length = (stop - start) / step
#     parameters = []
#     return_info_temp = copy.deepcopy(return_info)
#     for info in return_info_temp:
#         if info not in mode:
#             return_info.remove(info)
#             print "Warning: The required infomation %s is not contained in file %s" %(info, filename)
#             continue
#         if info == "force":
#             force = np.zeros((data_length, num_atom, 3), dtype="double")
#             parameters.append(force)
#         elif info == "energy":
#             energy = np.zeros((data_length, num_atom), dtype="double")
#             parameters.append(energy)
#         elif info == "stress":
#             stress = np.zeros((data_length, num_atom, 3, 3), dtype="double")
#             parameters.append(stress)
#         elif info == 'heatflux':
#             heatflux = np.zeros((data_length, 3), dtype="double")
#             parameters.append(heatflux)
#         else:
#             return_info.remove(info)
#     input_file=file(filename,'r')
#     pass_lines(input_file, num_line_one_step * start) # goes to the first step
#     pass_lines(input_file, num_mode_line[0]) # goes to the first MD line
#     for record in range(data_length):
#         line_in_one_record = 0
#         for i,m in enumerate(mode): # m: mode; n: line; p: parameter
#             if m not in return_info: # pass by the redundant information
#                 line_in_one_record += pass_lines(input_file, num_mode_line[i+1]-num_mode_line[i])
#             else:
#                 line_in_one_record += pass_lines(input_file, 1) # pass by the infomation line
#                 if m == 'force':
#                     line_temp, p_temp = pass_lines(input_file, num_atom, return_lines=True)
#                     line_in_one_record += line_temp
#                     force[record]= np.array([np.fromstring(l, sep=" ") for l in p_temp])
#                 elif m == "energy":
#                     line_temp, p_temp = pass_lines(input_file, num_atom, return_lines=True)
#                     line_in_one_record += line_temp
#                     energy[record] = np.array(p_temp)
#                 elif m == "stress":
#                     line_temp, p_temp = pass_lines(input_file, num_atom, return_lines=True)
#                     line_in_one_record += line_temp
#                     stress[record] = np.array([np.fromstring(l, sep=" ").reshape(3,3) for l in p_temp])
#                 elif m == "heatflux":
#                     line_temp, p_temp = pass_lines(input_file, 1, return_lines=True)
#                     line_in_one_record += line_temp
#                     heatflux[record] = np.array([np.fromstring(l, sep=" ") for l in p_temp]).flatten()
#                 rest_lines = num_mode_line[i+1]-num_mode_line[i] - line_temp - 1
#                 line_in_one_record += pass_lines(input_file, rest_lines)
#         pass_lines(input_file, num_line_one_step - line_in_one_record) # pass by those uncollected information
#         pass_lines(input_file, num_line_one_step * (step-1))
#     return parameters, return_info
#
# @timeit
# def read_xyz_fe_file3(filename, num_atom, return_info=None, step_range=slice(None)):
#     if return_info == None:
#         return_info = ['force', 'energy', 'stress', 'heatflux']
#     total_lines = file_len(filename)
#     mode, num_mode_line = get_one_step_number_of_lines(filename)
#     num_line_one_step = num_mode_line[-1] - num_mode_line[0]
#     start, stop, info = (step_range.start, step_range.stop, step_range.step)
#     start = start if start is not None else 0
#     info = info if info is not None else 1
#     stop = stop if stop is not None else total_lines / num_line_one_step
#     if not check_file_length(total_lines, num_line_one_step, start, stop, info):
#         return None
#
#     data_length = (stop - start) / info
#     parameters = []
#     return_info_temp = copy.deepcopy(return_info)
#     for info in return_info_temp:
#         if info not in mode:
#             return_info.remove(info)
#             print "Warning: The required infomation %s is not contained in file %s" %(info, filename)
#             continue
#         if info == "force":
#             force = np.zeros((data_length, num_atom, 3), dtype="double")
#             parameters.append(force)
#         elif info == "energy":
#             energy = np.zeros((data_length, num_atom), dtype="double")
#             parameters.append(energy)
#         elif info == "stress":
#             stress = np.zeros((data_length, num_atom, 3, 3), dtype="double")
#             parameters.append(stress)
#         elif info == 'heatflux':
#             heatflux = np.zeros((data_length, 3), dtype="double")
#             parameters.append(heatflux)
#         else:
#             return_info.remove(info)
#     atom_energies = np.zeros((data_length, num_atom))
#     atom_stresses = np.zeros((data_length, num_atom, 3, 3), dtype="double")
#     atom_forces = np.zeros((data_length, num_atom,3))
#     for i, info in enumerate(return_info):
#         if info == "stress":
#             p = subprocess.Popen(['sed', '-n', '-e', "\'/stress/, +%d{/stress/d; p}\'" %num_atom],
#                                  stdout=subprocess.PIPE,
#                                  stderr=subprocess.PIPE)
#             result, err = p.communicate()
#             if p.returncode != 0:
#                 raise IOError(err)
#             atom_stresses = np.fromstring(result).reshape(num_atom, 3,3)
#         if info == "force":
#             p = subprocess.Popen(['sed', '-n', '-e', "/Forces/, +%d{/Forces/d; p}" %num_atom, filename],
#                                  stdout=subprocess.PIPE,
#                                  stderr=subprocess.PIPE)
#             result, err = p.communicate()
#             if p.returncode != 0:
#                 raise IOError(err)
#             atom_forces = np.array(result.split()).reshape(data_length, num_atom, 3)
#     return 1


def read_xyz_fe_file(filename, num_atom, step_range=slice(None)):
    "unit of force: Ha/Bohr (au), energy: Hartree (au)"
    mode, num_mode_line = get_one_step_number_of_lines(filename)
    input_file=file(filename)
    file_all = input_file.read()
    file_list_all=file_all.split('MD step:')[1:]
    try:
        file_list=file_list_all[step_range]
    except IndexError:
        return None
    data_length = len(file_list)
    parameters = {}
    for info in mode:
        if info == "force":
            force = np.zeros((data_length, num_atom, 3), dtype="double")
            parameters.setdefault("forces", force)
        elif info == "energy":
            energy = np.zeros((data_length, num_atom), dtype="double")
            parameters.setdefault("energies", energy)
        elif info == "stress":
            stress = np.zeros((data_length, num_atom, 3, 3), dtype="double")
            parameters.setdefault("stresses",stress)
        elif info == 'heatflux':
            heatflux = np.zeros((data_length, 3), dtype="double")
            parameters.setdefault("heatfluxes",heatflux)

    for i, step in enumerate(file_list):
        if "stress" in mode:
            stress_lines = step.split("Atom resolved stress tensors")[-1].strip().split("\n")[:num_atom]
            stress = np.array([np.fromstring(s, sep=" ", dtype="double") for s in stress_lines])
            parameters['stresses'][i] = stress.reshape(num_atom, 3,3)
        if "energy" in mode:
            energies = step[step.rfind("Atom resolved total energies"):].split("\n")[1:num_atom+1]
            parameters['energies'][i] = np.array([line.split()[1] for line in energies], dtype="double") * au_energy
        if "force" in mode:
            forces = step[step.rfind("Total Forces"):].split("\n")[1:num_atom+1]
            parameters['forces'][i] = np.array([line.split() for line in forces], dtype="double") * au_force
        if "heatflux" in mode:
            heatflux = step[step.rfind("Heat Flux"):].split("\n")[1]
            parameters['heatfluxes'][i] = np.array(heatflux, dtype="double")
    return parameters

def write_md_to_hdf5(filename="mdinfo.hdf5",
                     coordinate=None,
                     velocity=None,
                     type=None,
                     force=None,
                     energy=None,
                     stress=None,
                     volume=None,
                     time_step=None,
                     heat_flux=None):
    import h5py
    o=h5py.File(filename,"w")
    for name, para in zip(('velocities', 'coordinates', 'atom_types', 'time_step', 'forces', 'volumes', 'energies', 'stress', 'heat_fluxes'),
                          (velocity,      coordinate,    type,         time_step,   force,    volume,   energy,   stress,   heat_flux)):
        if para is not None:
            o.create_dataset(name, data=para)
    o.close()


def read_md_from_hdf5(filename):
    import h5py
    i=h5py.File(filename,'r')
    parameters = {}
    for key in i.keys():
        parameters.setdefault(key, i[key].value)
    i.close()
    return parameters


def read_md_from_vasprun(vasp_filename):
    from lxml import etree
    f = open(vasp_filename, "r")
    lines = f.read()
    if lines.strip()[-100].find("</modeling>") == -1:
        if lines.strip()[-8:] == "<scstep>":
            lines += "</scstep>\n"
        lines += "</calculation>\n"
        lines += "</modeling>\n"
    f.close()
    f = StringIO.StringIO(lines)
    vasprun = etree.iterparse(f, tag='calculation')
    forces = []
    for event, element in vasprun:
        force = []
        for ele in element.iterfind('varray'):
            if ele.get("name") == 'forces':
                for v in ele.xpath('./v'):
                    force.append([float(x) for x in v.text.split()])
            forces.append(force)
            break
    parameter = {'forces': np.array(forces, dtype="double")}
    return parameter

if __name__=="__main__":
    # read_xyz_fe_file3(filename="heat_flux0.out", num_atom=64, return_info=['force'], step_range=slice(None))
    # read_xyz_fe_file2(filename="heat_flux0.out", num_atom=64, return_info=['force'], step_range=slice(None))
    # read_xyz_fe_file2(filename="heat_flux0.out", num_atom=64, return_info=['force'], step_range=slice(None))
    parameters, mode = read_xyz_fe_file(filename="heat_flux0.out", num_atom=64, return_info=['force', 'energy'], step_range=slice(None))

    for i, p in zip(mode, parameters):
        print "The returned information is: %s"%i
        print "shape of parameters:", p.shape
    # md=MD(fileformat="x",filename="cutstep_4000.xyz")
    
