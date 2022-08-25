#!/usr/bin/env python
"""
- Read a PDB file.
- Align the corresponding contact map to the given reference contact map
- Get the fragments from the pdb with the sequence attributed from the contact
map alignment
- Use modeller to get the full atom models
- Refine the fragements into the density using phenix.real_space_refine
"""
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-12-22 13:34:34 (UTC+0100)

import sys
sys.path.append('/home/bougui/source/modeller_repair_protein')
import repair_protein
import os
import numpy
import map_align
import all_atoms
import optimizer
import tempfile
import pdbalign

PDB = sys.argv[1]
CMAP = sys.argv[2]
SEQUENCE = sys.argv[3]
GAP = int(sys.argv[4]) # Where to split in fragment (Gap in number of residues)
THRESHOLD = float(sys.argv[5]) # Distance threshold for contact maps

atom_types, aa_list, chain_ids, resids, coords, _ = all_atoms.read_pdb(PDB)
selection = (numpy.asarray(atom_types) == 'CA')
coords = coords[selection]
cmap = map_align.read_gremlin(CMAP)
aln = map_align.map_align(coords, cmap, fasta_file=SEQUENCE,
                          threshold=THRESHOLD)
resids = numpy.asarray(aln.alignment.values()) + 1
# split the pdb into fragments given the alignment:
seq = numpy.asarray(aln.sequence)
# split where there is a X:
split_X = numpy.zeros(len(seq), dtype=bool)
for i in range(1, len(seq)-1):
    if seq[i+1] == 'X' and seq[i] != 'X':
        split_X[i+1] = True
    if seq[i-1] == 'X' and seq[i] != 'X':
        split_X[i] = True
selection = (seq != 'X')
coords_X = coords[~selection] # Unattributed coordinates
fragments_X = numpy.split(coords_X, numpy.where(split_X[~selection])[0])
split_X = list(split_X[selection])
seq = seq[selection]
coords = coords[selection]
# Split where there is a GAP residues long gap:
gap = GAP
split_pos = list(numpy.where(numpy.diff(resids) > gap)[0]+1)
split_pos.extend(list(numpy.where(split_X)[0]))
split_pos.sort()
fragments = numpy.split(coords, split_pos)
resids_ = numpy.split(resids, split_pos)
seq_ = numpy.split(seq, split_pos)
i = 0
for fragment in fragments:
    if len(fragment) > 4:
        OUTFILENAME = "fragment_%d.pdb"%i
        optimizer.write_pdb(fragment, resids=resids_[i],
                            outfilename=OUTFILENAME, sequence=seq_[i])
    i += 1
    #modeller_outname = '%s/model_%d.pdb'%(tempdir, i)
    #repair_protein.repair_protein(OUTFILENAME,
    #                              outpdb=modeller_outname)
    #aligner = pdbalign.PDBalign(OUTFILENAME, modeller_outname)
    #aligner.align()
    #aligner.write_pdb(outfilename=modeller_outname)
    #all_atoms.real_space_refine(modeller_outname, MRC, 5., 2)
for fragment in fragments_X:
    if len(fragment) > 4:
        optimizer.write_pdb(fragment, outfilename="fragment_%d.pdb"%i)
    i += 1
