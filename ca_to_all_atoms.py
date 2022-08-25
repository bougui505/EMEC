#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import map_align
import DSSP
import optimizer
import all_atoms
import shutil
import numpy
import ConfigParser
import sys

Config = ConfigParser.ConfigParser()
Config.read(sys.argv[1])
nc_file = Config.get('EM maps', 'nc')
mrc_file = Config.get('EM maps', 'mrc')
resolution = float(Config.get('EM maps', 'resolution'))
gremlin_file = Config.get('Structure files', 'gremlin_file')
fasta_file = Config.get('Structure files', 'fasta_file')
ca_trace = Config.get('Structure files', 'ca_trace')
n_iter = int(Config.get('algo', 'n_iter'))
ss_restraints = bool(int(Config.get('algo', 'ss_restraints')))

_, _, _, _, coords, _ = all_atoms.read_pdb(ca_trace)

em_level = -numpy.inf
malign = map_align.MapAlign([coords, ], nc_file, em_level, gremlin_file,
                            fasta_file)
malign.coords_best = coords
malign.aln = map_align.map_align(malign.coords, malign.gmap,
                                 fasta_file=malign.fasta_file)
# Set up the class to compute the secondary structure attribution
dssp = DSSP.DSSP(None, nc_file, em_level)

for i in range(n_iter):
    print "%d/%d"%(i+1, n_iter)
    malign.optimize_ca()
    optimizer.write_pdb(malign.coords, outfilename='fit.pdb', sequence=malign.aln.sequence)
    if dssp.pdbfilename is not None and ss_restraints:
        # Get the secondary structures restraints for modeller
        alpha, strand = dssp.get_ss_restraints('fit.pdb')
        if alpha is not None:
            print "Alpha helix restraints: %s"%alpha
        if strand is not None:
            print "Beta strand restraints: %s"%strand
    else:
        alpha, strand = None, None
    aa = all_atoms.AllAtoms('fit.pdb', 'map_align.fasta', nc_file, em_level,
                            mrc=mrc_file, resolution=resolution, alpha=alpha,
                            strand=strand)
    # Set the pdbfile in dssp to compute secondary structure attribution of the EM map
    dssp.pdbfilename = 'modeller_out.pdb'
    shutil.copy('modeller_out.pdb', 'mod_%d.pdb'%i)
    malign.coords_best = aa.ca_trace
    # map alignment after modeller optimization for all atoms
    malign.aln = map_align.map_align(aa.cmap, malign.gmap,
                                     fasta_file=malign.fasta_file)


