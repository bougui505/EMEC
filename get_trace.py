#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-02-27 17:01:14 (UTC+0100)

import skeleton
import numpy
import sys
import optimizer
import warnings
warnings.filterwarnings("ignore") # Do not display python warnings

emd = sys.argv[1]
level = float(sys.argv[2])

skl = skeleton.Skeleton(emd, level)

for i, chain in enumerate(skl.chains):
    optimizer.write_minimum_spanning_tree(chain.coords, chain.topology, outfile='mstree_%d.pdb'%i)
    optimizer.write_minimum_spanning_tree(chain.coords, chain.backbone, outfile='backbone_%d.pdb'%i)
    optimizer.write_minimum_spanning_tree(chain.coords, chain.sidechains, outfile='sidechains_%d.pdb'%i)
    optimizer.write_minimum_spanning_tree(chain.coords, chain.protein, outfile='protein_%d.pdb'%i)
    coords_CA = numpy.copy(chain.coords_CA)
    optimizer.write_pdb(coords_CA, outfilename='ca_trace_%d.pdb'%i)
    numpy.save('ca_trace_%d.npy'%i, coords_CA)
