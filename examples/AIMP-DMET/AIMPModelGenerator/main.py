# utilies script: write structure file
import numpy as np
import yaml
import sys, getopt
#from distutils.command.config import config
from fitting_ewald.neighborTools import neighbors
from fitting_ewald.potential_fitting import PotentialFitOnlyCharges

opts, args = getopt.getopt(sys.argv[1:], "i:")
for op, value in opts:
    if op == "-i": input_file = value

def write_xyz(nbs: neighbors, filename="mol.xyz"):
    natoms = nbs.get_number_of_neighbors()
    indices = nbs.get_neighbors()[0]
    cell_symbols = nbs.mol.get_chemical_symbols()
    labels = [cell_symbols[index] for index in indices]
    crds = nbs.get_cartesian_coordinates(origin_shifted=True)
    with open(filename, "w") as f:
        print(" ", natoms, file=f)
        print(" structure file generated manunally.", file=f)
        for i, label in enumerate(labels):
            print(f"  {label:3s}  %15.7f %15.7f %15.7f" % tuple(crds[i]), file=f)
    return

f = open(input_file, "r")
configdict = yaml.safe_load(f)

poscarfile = configdict["poscarfile"]
rCluster = configdict["rCluster"]
rAIMP = configdict["rAIMP"]
rChgs = configdict["rChgs"]
rSurface = configdict["rSurface"]
cAtom = configdict["cAtom"]
cAtomIndex = configdict["cAtomIndex"]
num_sites = configdict["num_sites"]

cluster_nbs = neighbors(poscarfile, cAtom=cAtom, cAtomIndex=cAtomIndex, rCut=rCluster, sort=True)
aimp_nbs = neighbors(poscarfile, cAtom=cAtom, cAtomIndex=cAtomIndex, rCore=rCluster, rCut=rAIMP, sort=True)
rawChgs_nbs = neighbors(poscarfile, cAtom=cAtom, cAtomIndex=cAtomIndex, rCore=rAIMP, rCut=rChgs, sort=True)
pot_fit = PotentialFitOnlyCharges(poscarfile, cAtom=cAtom, cAtomIndex=cAtomIndex, rCut=rChgs, rSurface=rSurface, num_sites=num_sites)
pot_fit.run_fit()
pot_fit.show_res()

write_xyz(cluster_nbs, "cluster.xyz")
write_xyz(aimp_nbs, "aimp.xyz")
write_xyz(rawChgs_nbs, "rawChgs.xyz")
write_xyz(pot_fit.surface_neighbors, "surfChgs.xyz")
np.savetxt("rawCharges.dat", rawChgs_nbs.get_charge_list())
np.savetxt("surfaceCharges.dat", pot_fit.surf_chgs)
