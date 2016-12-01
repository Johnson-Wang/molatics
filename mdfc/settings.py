__author__ = 'xinjiang'

import sys
import numpy as np
from realmd.information import warning

def fracval(frac):
    if frac.find('/') == -1:
        return float(frac)
    else:
        x = frac.split('/')
        return float(x[0]) / float(x[1])

def bool2string(bool_num):
    if not type(bool_num) is bool:
        return None
    if bool_num ==True:
        return ".true."
    else:
        return ".false."

class Settings:
    def __init__(self):
        self._is_tensor_symmetry = False
        self._is_translational_invariance = False
        self._is_rotational_invariance = False
        self._is_symmetry = True
        self._primitive_matrix = np.eye(3, dtype=float)
        self._run_mode = None
        self._cutoff_radius = None
        self._cutoff_disp = None
        self._supercell_matrix = None
        self._supercell_matrix_orig = None
        self._is_time_symmetry = True
        self._is_hdf5 = False
        self._is_fc3 = False
        self._is_fc2 = False
        self._read_fc2 = None
        self._read_fc3 = None
        self._t = 300
        self._cutoff_residual_force = 1e-8 #eV/A^2
        self._precision = 1e-8
        self._predict_count = 1
        self._interface = "dftb"
        self._step_range=slice(None)
        self._file_format="x"
        self._coord_filename = "geo_end.xyz"
        self._force_filename = 'heat_flux.out'
        self._is_convert_input = False
        self._divide=1
        self._is_disperse=False

    def set_run_mode(self, run_mode):
        self._run_mode = run_mode

    def get_run_mode(self):
        return self._run_mode

    def set_supercell_matrix(self, matrix):
        self._supercell_matrix = matrix

    def get_supercell_matrix(self):
        return self._supercell_matrix

    def set_supercell_matrix_orig(self, matrix):
        self._supercell_matrix_orig = matrix

    def get_supercell_matrix_orig(self):
        return self._supercell_matrix_orig

    def set_time_symmetry(self, time_symmetry=True):
        self._is_time_symmetry = time_symmetry

    def get_is_disperse(self):
        return self._is_disperse

    def set_is_disperse(self, is_disperse):
        self._is_disperse = is_disperse

    def get_time_symmetry(self):
        return self._is_time_symmetry

    def set_primitive_matrix(self, primitive_matrix):
        self._primitive_matrix = primitive_matrix

    def get_primitive_matrix(self):
        return self._primitive_matrix

    def set_is_tensor_symmetry(self, is_tensor_symmetry):
        self._is_tensor_symmetry = is_tensor_symmetry

    def get_is_tensor_symmetry(self):
        return self._is_tensor_symmetry

    def set_is_fc3(self, is_fc3):
        self._is_fc3 = is_fc3

    def get_is_fc3(self):
        return self._is_fc3

    def set_is_fc2(self, is_fc2):
        self._is_fc2 = is_fc2

    def get_is_fc2(self):
        return self._is_fc2

    def set_read_fc3(self, read_fc3):
        self._read_fc3 = read_fc3

    def get_read_fc3(self):
        return self._read_fc3

    def set_read_fc2(self, read_fc2):
        self._read_fc2 = read_fc2

    def get_read_fc2(self):
        return self._read_fc2

    def set_is_translational_invariance(self, is_translational_invariance):
        self._is_translational_invariance = is_translational_invariance

    def get_is_translational_invariance(self):
        return self._is_translational_invariance

    def set_is_rotational_invariance(self, is_rotational_invariance):
        self._is_rotational_invariance = is_rotational_invariance

    def get_is_rotational_invariance(self):
        return self._is_rotational_invariance

    def set_is_symmetry(self, is_symmetry):
        self._is_symmetry = is_symmetry

    def get_is_symmetry(self):
        return self._is_symmetry

    def set_temperature(self, t):
        self._t = t

    def get_temperature(self):
        return self._t

    def set_cutoff_residual_force(self, cutoff_residual_force):
        self._cutoff_residual_force = cutoff_residual_force

    def get_cutoff_residual_force(self):
        return self._cutoff_residual_force

    def set_precision(self, precision):
        self._precision = precision

    def get_precision(self):
        return self._precision

    def set_predict_count(self, predict_coun):
        self._predict_count = predict_coun

    def get_predict_coun(self):
        return self._predict_count

    def set_cutoff_radius(self, cutoff_radius):
        self._cutoff_radius = cutoff_radius

    def get_cutoff_radius(self):
        return self._cutoff_radius

    def set_cutoff_disp(self, cutoff_disp):
        self._cutoff_disp = cutoff_disp

    def get_cutoff_disp(self):
        return self._cutoff_disp

    def set_interface(self, interface):
        self._interface = interface

    def get_interface(self):
        return self._interface

    def set_is_hdf5(self, is_hdf5):
        self._is_hdf5 = is_hdf5

    def get_is_hdf5(self):
        return self._is_hdf5

    def set_file_format(self, file_format):
        self._file_format = file_format

    def get_file_format(self):
        return self._file_format

    def set_step_range(self, step_range):
        self._step_range = step_range

    def get_step_range(self):
        return self._step_range

    def set_coord_filename(self, coord_filename):
        self._coord_filename = coord_filename

    def get_coord_filename(self):
        return self._coord_filename

    def set_force_filename(self, force_filename):
        self._force_filename = force_filename

    def get_force_filename(self):
        return self._force_filename

    def set_is_convert_input(self, is_convert_input):
        self._is_convert_input = is_convert_input

    def get_is_convert_input(self):
        return self._is_convert_input

    def set_divide(self, divide):
        self._divide=divide

    def get_divide(self):
        return self._divide

class ConfParser:
    def __init__(self, filename=None, options=None, option_list=None):
        self._confs = {}
        self._parameters = {}
        self._options = options
        self._option_list = option_list

        if filename is not None:
            self.read_file(filename) # store data in self._confs
        if (options is not None) and (option_list is not None):
            self.read_options() # store data in self._confs
        self.parse_conf() # self.parameters[key] = val
        self.parameter_coupling()
        self._settings = Settings()
        self.set_settings()


    def get_settings(self):
        return self._settings

    def setting_error(self, message):
        print message
        print "Please check the setting tags and options."
        sys.exit(1)

    def set_settings(self):
        params = self._parameters
        # Primitive cell shape
        if params.has_key('primitive_axis'):
            self._settings.set_primitive_matrix(params['primitive_axis'])

        if params.has_key('supercell_matrix'):
            self._settings.set_supercell_matrix(params['supercell_matrix'])

        if params.has_key('supercell_matrix_orig'):
            self._settings.set_supercell_matrix_orig(params['supercell_matrix_orig'])

        # Is crystal symmetry searched?
        if params.has_key('is_symmetry'):
            self._settings.set_is_symmetry(params['is_symmetry'])

        # Is translational invariance ?
        if params.has_key('is_translational'):
            self._settings.set_is_translational_invariance(
                params['is_translational'])

        # Is rotational invariance ?
        if params.has_key('is_rotational'):
            self._settings.set_is_rotational_invariance(params['is_rotational'])

        # Is force constants symmetry forced?
        if params.has_key('is_tensor_symmetry'):
            self._settings.set_is_tensor_symmetry(params['is_tensor_symmetry'])

        if params.has_key('temperature'):
            self._settings.set_temperature(params['temperature'])

        if params.has_key('cutoff_residual_force'):
            self._settings.set_cutoff_residual_force(params['cutoff_residual_force'])

        if params.has_key('precision'):
            self._settings.set_precision(params['precision'])

        if params.has_key('cutoff_radius'):
            self._settings.set_cutoff_radius(params['cutoff_radius'])

        if params.has_key('cutoff_disp'):
            self._settings.set_cutoff_disp(params['cutoff_disp'])

        if params.has_key('interface'):
            self._settings.set_interface(params['interface'])

        if params.has_key('is_hdf5'):
            self._settings.set_is_hdf5(params['is_hdf5'])

        if params.has_key('is_fc3'):
            self._settings.set_is_fc3(params['is_fc3'])

        if params.has_key('is_fc2'):
            self._settings.set_is_fc2(params['is_fc2'])

        if params.has_key('is_disperse'):
            self._settings.set_is_disperse(params['is_disperse'])

        if params.has_key('read_fc3'):
            self._settings.set_read_fc3(params['read_fc3'])

        if params.has_key('read_fc2'):
            self._settings.set_read_fc2(params['read_fc2'])

        if params.has_key('step_range'):
            self._settings.set_step_range(params['step_range'])

        if params.has_key('file_format'):
            self._settings.set_file_format(params['file_format'])

        if params.has_key('coord_filename'):
            self._settings.set_coord_filename(params['coord_filename'])

        if params.has_key('force_filename'):
            self._settings.set_force_filename(params['force_filename'])

        if params.has_key('is_convert_input'):
            self._settings.set_is_convert_input(params['is_convert_input'])

        if params.has_key('divide'):
            self._settings.set_divide(params['divide'])

        if params.has_key('predict_count'):
            self._settings.set_predict_count(params['predict_count'])


    def read_file(self, filename):
        file = open(filename, 'r')
        confs = self._confs
        is_continue = False
        for line in file:
            if line.strip() == '':
                is_continue = False
                continue

            if line.strip()[0] == '#':
                is_continue = False
                continue

            if is_continue:
                confs[left] += line.strip()
                confs[left] = confs[left].replace('+++', ' ')
                is_continue = False

            if line.find('=') != -1:
                left, right = [x.strip().lower() for x in line.split('=')]
                right = right.split("#")[0].strip()
                if left == 'band_labels':
                    right = [x.strip() for x in line.split('=')][1]
                confs[left] = right

            if line.find('+++') != -1:
                is_continue = True

    def read_options(self):
        for opt in self._option_list:
            if opt.dest == 'primitive_axis':
                if self._options.primitive_axis:
                    self._confs['primitive_axis'] = self._options.primitive_axis

            if opt.dest == 'supercell_dimension':
                if self._options.supercell_dimension:
                    self._confs['dim'] = self._options.supercell_dimension

            if opt.dest == 'supercell_dimension_orig':
                if self._options.supercell_dimension_orig:
                    self._confs['rdim'] = self._options.supercell_dimension_orig

            if opt.dest == 'is_nosym':
                if self._options.is_nosym:
                    self._confs['symmetry'] = '.false.'

            if opt.dest == 'temperature':
                if self._options.temperature:
                    self._confs['temperature'] = self._options.temperature

            if opt.dest == 'cutoff_radius':
                if self._options.cutoff_radius:
                    self._confs['cutoff_radius'] = self._options.cutoff_radius

            if opt.dest == 'cutoff_disp':
                if self._options.cutoff_disp:
                    self._confs['cutoff_disp'] = self._options.cutoff_disp

            if opt.dest == 'is_rotational':
                if self._options.is_rotational:
                    self._confs['is_rotational'] = '.true.'

            if opt.dest == 'is_translational':
                if self._options.is_translational:
                    self._confs['is_translational'] = '.true.'

            if opt.dest == 'interface':
                if self._options.interface:
                    self._confs['interface'] = self._options.interface

            if opt.dest == 'is_hdf5':
                if self._options.is_hdf5:
                    self._confs['is_hdf5'] = '.true.'

            if opt.dest == 'is_fc3':
                if self._options.is_fc3:
                    self._confs['is_fc3'] = '.true.'

            if opt.dest == 'is_disperse':
                if self._options.is_disperse:
                    self._confs['is_disperse'] = '.true.'

            if opt.dest == 'is_fc2':
                if self._options.is_fc2:
                    self._confs['is_fc2'] = '.true.'

            if opt.dest == 'read_fc2':
                if self._options.read_fc2:
                    self._confs['read_fc2'] = self._options.read_fc2

            if opt.dest == 'read_fc3':
                if self._options.read_fc3:
                    self._confs['read_fc3'] = self._options.read_fc3

            if opt.dest=="step_range":
                ran=self._options.step_range.strip(" \'\"")
                self._confs["step_range"]=ran

            if opt.dest=="file_format":
                self._confs["file_format"]=self._options.file_format.strip(" \'\"")

            if opt.dest=="coord_filename":
                self._confs["coord_filename"]=self._options.coord_filename

            if opt.dest=="force_filename":
                self._confs["force_filename"]=self._options.force_filename

            if opt.dest=="is_convert_input":
                convert = bool2string(self._options.is_convert_input)
                self._confs['is_convert_input'] = convert

            if opt.dest=="divide":
                self._confs["divide"]=int(self._options.divide)

            if opt.dest=="cutoff_residual_force":
                self._confs["cutoff_residual_force"]=float(self._options.cutoff_residual_force)

            if opt.dest=="precision":
                self._confs["precision"]=float(self._options.precision)

            if opt.dest=="predict_count":
                self._confs["predict_count"]=int(self._options.predict_count)

    def parse_conf(self):
        confs = self._confs

        for conf_key in confs.keys():
            if conf_key == 'dim':
                matrix = [int(x) for x in confs['dim'].split()]
                if len(matrix) == 9:
                    matrix = np.array(matrix).reshape(3, 3)
                elif len(matrix) == 3:
                    matrix = np.diag(matrix)
                else:
                    self.setting_error(
                        "Number of elements of DIM tag has to be 3 or 9.")

                if matrix.shape == (3, 3):
                    if np.linalg.det(matrix) < 1:
                        self.setting_error(
                            'Determinant of supercell matrix has to be positive.')
                    else:
                        self.set_parameter('supercell_matrix', matrix)

            if conf_key == 'rdim':
                matrix = [int(x) for x in confs['rdim'].split()]
                if len(matrix) == 9:
                    matrix = np.array(matrix).reshape(3, 3)
                elif len(matrix) == 3:
                    matrix = np.diag(matrix)
                else:
                    self.setting_error(
                        "Number of elements of DIM tag has to be 3 or 9.")

                if matrix.shape == (3, 3):
                    if np.linalg.det(matrix) < 1:
                        self.setting_error(
                            'Determinant of supercell matrix has to be positive.')
                    else:
                        self.set_parameter('supercell_matrix_orig', matrix)

            if conf_key == 'primitive_axis':
                if not len(confs['primitive_axis'].split()) == 9:
                    self.setting_error(
                        "Number of elements in PRIMITIVE_AXIS has to be 9.")
                p_axis = []
                for x in confs['primitive_axis'].split():
                    p_axis.append(fracval(x))
                p_axis = np.array(p_axis).reshape(3,3)
                if np.linalg.det(p_axis) < 1e-8:
                    self.setting_error(
                        "PRIMITIVE_AXIS has to have positive determinant.")
                self.set_parameter('primitive_axis', p_axis)

            if conf_key == 'symmetry':
                if confs['symmetry'] == '.false.':
                    self.set_parameter('is_symmetry', False)
                    self.set_parameter('is_mesh_symmetry', False)

            if conf_key == 'is_translational':
                if confs['is_translational'] == ".true.":
                    self.set_parameter('is_translational',True)

            if conf_key == 'is_rotational':
                if confs['is_rotational'] == ".true.":
                    self.set_parameter('is_rotational', True)

            if conf_key == 'tensor_symmetry':
                if confs['tensor_symmetry'] == '.true.':
                    self.set_parameter('is_tensor_symmetry', True)

            if conf_key == 'temperature':
                val = float(confs['temperature'])
                self.set_parameter('temperature', val)

            if conf_key == 'cutoff_radius':
                val = map(float, confs['cutoff_radius'].strip().replace(",", " ").split())
                self.set_parameter('cutoff_radius', val)

            if conf_key == 'cutoff_disp':
                val = map(float, confs['cutoff_disp'].strip().replace(",", " ").split())
                if len(val) == 0:
                    val = None
                elif len(val) == 1:
                    val = val[0]
                elif len(val) == 3:
                    val = np.array(val)
                else:
                    print "The cutoff_disp is set incorrectly!"
                    sys.exit(1)
                self.set_parameter('cutoff_disp', val)

            if conf_key == 'cutoff_residual_force':
                val = float(confs['cutoff_residual_force'])
                self.set_parameter('cutoff_residual_force', val)

            if conf_key == 'precision':
                val = float(confs['precision'])
                self.set_parameter('precision', val)

            if conf_key == 'interface':
                val = float(confs['interface'])
                self.set_parameter('interface', val)

            if conf_key == 'is_hdf5':
                if confs['is_hdf5'] == '.true.':
                    self.set_parameter('is_hdf5', True)

            if conf_key == 'is_fc3':
                if confs['is_fc3'] == '.true.':
                    self.set_parameter('is_fc3', True)

            if conf_key == 'is_disperse':
                if confs['is_disperse'] == '.true.':
                    self.set_parameter('is_disperse', True)

            if conf_key == 'is_fc2':
                if confs['is_fc2'] == '.true.':
                    self.set_parameter('is_fc2', True)

            if conf_key == 'read_fc2':
                self.set_parameter('read_fc2', confs['read_fc2'])

            if conf_key == 'read_fc3':
                self.set_parameter('read_fc3', confs['read_fc3'])

            if conf_key == "step_range":
                ran=confs["step_range"]
                if ran=="":
                    self.step_range=slice(None, None, None)
                elif ran.count(":")>0:
                    exec "s=np.s_[%s]"%ran
                    self.set_parameter('step_range', s)
                else:
                    print "Wrong format for step_range"
                    sys.exit(1)

            if conf_key == 'file_format':
                self.set_parameter('file_format', confs["file_format"][0])

            if conf_key == 'force_filename':
                self.set_parameter('force_filename', confs["force_filename"])

            if conf_key == "coord_filename":
                self.set_parameter('coord_filename', confs['coord_filename'])

            if conf_key == "divide":
                self.set_parameter('divide', confs['divide'])

            if conf_key == "predict_count":
                self.set_parameter('predict_count', confs['predict_count'])

            if conf_key == "is_convert_input":
                if confs['is_convert_input'] == ".true.":
                    self.set_parameter('is_convert_input', True)

    def set_parameter(self, key, val):
        self._parameters[key] = val

    def parameter_coupling(self):
        parameters = self._parameters
        if parameters.has_key("coord_filename") and parameters.has_key("file_format"):
            coord_filename = parameters['coord_filename']
            file_format = parameters['file_format']
            if coord_filename is not None:
                if coord_filename.split(".")[-1]=="xyz":
                    if file_format != "x":
                        warning("xyz file format detected, format converted forcibly!")
                        self.set_parameter('file_format', "x")
                elif coord_filename.split(".")[-1]=="hdf5":
                    if file_format != "h":
                        warning("hdf5 file format detected, format converted forcibly!")
                        self.set_parameter('file_format', 'h')
                elif coord_filename == "XDATCAR":
                    if file_format != "v":
                        warning("vasp file format detected, format converted forcibly!")
                        self.set_parameter('file_format', "v")

