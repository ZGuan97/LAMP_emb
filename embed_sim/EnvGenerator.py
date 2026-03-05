import numpy as np
Ang2Bohr = 1.8897259886
from pyscf import gto

def xyz_parser(filename):
    # Input: name of xyz file
    # Output: 
    #           Number of atom;
    #           number of information saved as dictionaries;
    #           string lines of coordinate
    file = open(filename, "r")
    file_lines = file.readlines()
    natom = eval(file_lines[0])
    dicts = []
    for i in range(natom):
        string = file_lines[2+i]
        atm, xstr, ystr, zstr = string.split()
        x, y, z = eval(xstr), eval(ystr), eval(zstr)
        adict = {"atom": atm, "coordinate": np.array([x, y, z])}
        dicts.append(adict)
    file.close()
    return natom, dicts, file_lines[2:]

def xyz2coords(filename): 
    # Only storing information of coordinates
    # used for generating point charge layers

    if filename[-4:] == ".xyz":
        with open(filename, 'r') as f:
            lines = f.readlines()
    else:
        lines = filename.split("\n")

    natom = eval(lines[0])
    alist = []
    for i in range(natom):
        string = lines[2+i]
        atm, xstr, ystr, zstr = string.split()
        x = eval(xstr) * Ang2Bohr
        y = eval(ystr) * Ang2Bohr
        z = eval(zstr) * Ang2Bohr
        alist.append((x, y, z))
    return np.array(alist)

def find_mid_atom(dicts, atom_name, natom):
    """
    Return the index of atom in the midst of crystal cell
    """
    min_x, min_y, min_z = dicts[0]["coordinate"]
    max_x, max_y, max_z = dicts[0]["coordinate"]
    for i in range(natom):
        x, y, z = dicts[i]["coordinate"]
        if x < min_x: min_x = x
        if x > max_x: max_x = x
        if y < min_y: min_y = y
        if y > max_y: max_y = y
        if z < min_z: min_z = z
        if z > max_z: max_z = z
    
    max_coord = np.array((max_x, max_y, max_z))
    min_coord = np.array((min_x, min_y, min_z))
    mid_coord = (max_coord + min_coord) / 2

    min_dist = np.linalg.norm(max_coord - min_coord)
    min_idx = 0
    for i in range(natom):
        if dicts[i]["atom"] == atom_name:
            normcoord = np.linalg.norm(dicts[i]["coordinate"] - mid_coord)
            if normcoord < min_dist:
                min_dist = normcoord
                min_idx = i
    return min_idx

def get_atm_str(filename, index):
    file = open(filename, "r")
    fileline = file.readlines()
    line = fileline[2 + index]
    file.close()
    return line

def cut_sphere(filename, radius, out_filename, atom=None, coord=None, chglst=None, chgdir=None):
    # Cut a sphere from a xyz file and save it into another.
    natm, dicts, filelines = xyz_parser(filename)
    newchglst = []
    if atom is not None:
        mid_atom_idx = find_mid_atom(dicts, atom, natm)
        mid_coord = dicts[mid_atom_idx]["coordinate"]
    else:
        mid_coord = np.array(coord)
    n = 0
    lines = []
    for i in range(natm):
        coord = dicts[i]["coordinate"]
        norm = np.linalg.norm(coord - mid_coord)
        if norm <= radius:
            n += 1
            atm = dicts[i]["atom"]
            x, y, z = coord - mid_coord
            xyz_line = "{}    {:.6f}    {:.6f}    {:.6f}\n".format(atm, x, y, z)
            lines.append(xyz_line)
            if chglst is not None:
                newchglst.append(chglst[i])
    out_file = open(out_filename, "a")
    out_file.write("{}\n".format(int(n)))
    out_file.write("{}\n".format(out_filename))
    for line in lines:
        out_file.write(line)
    out_file.close()
    if chgdir is not None:
        newchglst = np.array(newchglst)
        np.savetxt(chgdir, newchglst)

def get_outer_layer(filename, radius, out_filename, center_coord=(0, 0, 0), chglst=None, chgdir=None):
    natm_prev, dicts, filelines = xyz_parser(filename)
    natm = 0
    lines = []
    newchglst = []
    for i in range(natm_prev):
        if np.linalg.norm(dicts[i]["coordinate"] - np.array(center_coord)) > radius:
            natm += 1
            x, y, z = dicts[i]["coordinate"]
            atm = dicts[i]["atom"]
            xyz_line = "{}    {:.6f}    {:.6f}    {:.6f}\n".format(atm, x, y, z)
            lines.append(xyz_line)
            if chglst is not None:
                newchglst.append(chglst[i])
    out_file = open(out_filename, "a")
    out_file.write("{}\n".format(int(natm)))
    out_file.write("{}\n".format(out_filename))
    for line in lines: out_file.write(line)
    out_file.close()

    if chgdir is not None:
        newchglst = np.array(newchglst)
        np.savetxt(chgdir, newchglst)
    

def load_env_from_xyz(filename, basis_set, chargedict:dict, ecp={}):
    # load point charge environment from xyz file
    # Old version, has been anachronistic.
    mol2_lists = []
    natm, dicts, lines = xyz_parser(filename)
    count = 0
    for atm in chargedict.keys():
        mol2_lists.append([])
        for i in range(natm):
            atmstr = dicts[i]['atom']
            line = lines[i]
            if atmstr == atm:
                if isinstance(basis_set, str): bs = basis_set
                elif isinstance(basis_set, dict): bs = basis_set[atm]
                if isinstance(ecp, str): _ecp = {atm: ecp}
                elif isinstance(ecp, dict):
                    try: _ecp = {atm: ecp[atm]}
                    except: _ecp = {}

                mol = gto.M(atom=line, basis=bs, charge=chargedict[atm],
                    ecp=_ecp)
                mol.verbose = 0
                mol2_lists[count].append(mol)
        count += 1
    return mol2_lists

class XYZParser:
    def __init__(self, file=None):
        self.natom = 0
        self.atmlst = []
        self.coords = []
        if file is not None:
            with open(file, 'r') as f:
                lines = f.readlines()
            self.natom = eval(lines[0])
            for i in range(self.natom):
                string = lines[2+i]
                atm, xstr, ystr, zstr = string.split()
                x, y, z = eval(xstr), eval(ystr), eval(zstr)
                self.atmlst.append(atm)
                self.coords.append([x, y, z])
            self.coords = np.array(self.coords)
        
    def __mul__(self, alpha): # parser1 * a1 + parser2 * a2
        newp = XYZParser()
        newp.natom = self.natom
        newp.atmlst = self.atmlst
        newp.coords = self.coords * alpha
        return newp
    
    def __add__(self, other):
        newp = XYZParser()
        newp.natom = self.natom
        newp.atmlst = self.atmlst
        newp.coords = self.coords + other.coords
        return newp
    
    def __str__(self):
        astr = ''
        for i in range(self.natom):
            atm = self.atmlst[i]
            x, y, z = self.coords[i]
            astr += '{} {} {} {} ;'.format(atm, x, y, z)
        astr = astr[:-1]
        return astr

    def calc_dQ(self):
        from pyscf.data import elements
        qsq = 0
        for i in range(self.natom):
            atm = self.atmlst[i]
            atmidx = elements.ELEMENTS.index(atm)
            mas = elements.MASSES[atmidx]
            coord = self.coords[i]
            # print(coord)
            qsq += mas * np.einsum("i->", coord**2)
        dQ = np.sqrt(qsq)
        print("dQ = ", dQ)
        return dQ

