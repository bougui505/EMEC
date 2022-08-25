#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-08-31 13:59:08 (UTC+0200)

import sys
import numpy
import map_align
import all_atoms

pdbfilename = sys.argv[1]
outfilename = sys.argv[2]
atom_types, aa_list, chain_ids, resids, \
    coords, is_ca_trace = all_atoms.read_pdb(pdbfilename)
if not is_ca_trace:
    # Select only ca atoms
    coords = coords[numpy.asarray(atom_types)=="CA"]
cmap = map_align.get_cmap(coords)
map_align.write_cmap(cmap, outfilename=outfilename, gremlin_format=True)
