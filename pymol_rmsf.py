import numpy as np
from pymol import cmd
import sys
import os

import re

gro = sys.argv[1]
# rmsf = sys.argv[2]

cmd.load(gro, object=re.split('-|.', gro)[1])
mol = cmd.get_object_list()[0]
# inFile = open(rmsf, 'r')
#
# newB = []
# rid = []
# for line in inFile.readlines()[1:]:
#     newB.append(float(line.split()[6]))
#     rid.append(int(line.split()[1]))
# inFile.close()
#
# cmd.alter(mol, 'b=0.0')
#
# prid = []
# cmd.iterate('%s and name CA' % (mol), 'prid.append((resi))')
#
# for i, ri in enumerate(prid):
#     cmd.alter('%s and n. CA and resid %s' % (mol, str(ri)), "b=%s" % (str(newB[i])))

# print('rmsf max: ', max(newB))
# cmd.spectrum("b",'rainbow',"%s and name CA and (resid %s:%s)"%(mol,str(rid[0]),str(rid[-1])))
cmd.spectrum("b", 'white_red', "%s and name CA" % (mol), "0.00", "3.0")


# pymol pymol_rmsf_color.py -- 2b2x.pdb