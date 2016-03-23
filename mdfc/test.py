from phonopy.interface import vasp
import numpy as np
import _mdfc as mdfc
from phonopy.structure.cells import get_supercell
unitcell = vasp.read_vasp("POSCAR")
supercell=get_supercell(unitcell,2 * np.eye(3, dtype="intc"))
a, b = mdfc.test(supercell.get_scaled_positions(), supercell.get_cell())