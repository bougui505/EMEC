#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-02-20 15:50:56 (UTC+0100)

import numpy
import scipy.spatial.distance
import re
import em_fit
import optimizer
import sys
sys.path.append('/home/bougui/source/modeller_repair_protein')
import repair_protein
import Graph
import tempfile
import os
import subprocess

def read_pdb(pdbfilename):
    """
    Read the given PDB filename and returns the atom_types, aa_list and a
    numpy array with the coordinates
    """
    splitted_lines = []
    with open(pdbfilename) as pdbfile:
        for line in pdbfile:
            if line[:4] == 'ATOM':
                splitted_line = [line[:6], line[6:11], line[12:16], line[17:20], line[21], line[22:26], line[30:38], line[38:46], line[46:54]]
                splitted_lines.append(splitted_line)
    splitted_lines = numpy.asarray(splitted_lines)
    atom_types = [e.strip() for e in splitted_lines[:, 2]]
    aa_list = [e.strip() for e in splitted_lines[:, 3]]
    chain_ids = [e.strip() for e in splitted_lines[:, 4]]
    resids = [int(e) for e in splitted_lines[:, 5]]
    # Check if the given pdbfile is a CA trace only
    is_ca_trace = len(set(resids)) == len(resids)
    return atom_types, aa_list, chain_ids, resids, numpy.float_(splitted_lines[:, 6:9]), is_ca_trace

def get_cmap(pdbfile, threshold=5.):
    """
    Contact map for the all atom protein.
    The pairwise distance between all atoms of each residue is computed and
    the minimal distance between atoms of residue i and residue j is kept.
    • pdbfile: file name for the all atoms PDB file
    • threshold: distance threshold above which defined a contact
    """
    atom_types, aa_list, chain_ids, resids, coords, _ = read_pdb(pdbfile)
    pdist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))
    cmap = numpy.zeros((len(set(resids)),)*2)
    for pt_i, i in enumerate(numpy.unique(resids)):
        for pt_j, j in enumerate(numpy.unique(resids)):
            cmap[pt_i, pt_j] = pdist[resids == i][:, resids == j].min()
    cmap = numpy.int_(cmap <= threshold)
    return cmap

def real_space_refine(pdb, mrc, resolution, nproc):
    """
    Use phenix.real_space_refine to refine the given pdb to the given MRC file
    """
    fd, path = tempfile.mkstemp()
    with open(path, 'w') as conf_file:
        conf_file.write("nproc = %d\n"%nproc)
        conf_file.write("refinement {\n")
        conf_file.write("run = minimization_global+local_grid_search+morphing+simulated_annealing\n")
        conf_file.write("}\n")
        conf_file.write("pdb_interpretation {\n")
        conf_file.write("secondary_structure {\n")
        conf_file.write("enabled = True\n")
        conf_file.write("protein {\n")
        conf_file.write("enabled = True\n")
        conf_file.write("search_method = from_ca\n")
        conf_file.write("remove_outliers = False\n")
        conf_file.write("}\n")
        conf_file.write("}\n")
        conf_file.write("}\n")
    args = ['phenix.real_space_refine', pdb, mrc, 'resolution=%.2f'%resolution,
            path]
    p = subprocess.Popen(args)
    p.wait()
    os.close(fd)

class AllAtoms(object):
    """
    Build an all atom protein fitting the EM density from a CA trace
    """
    def __init__(self, pdbfile, emd, level, mrc, resolution, alpha=None,
                 strand=None, basename=None, fasta=None):
        """
        • pdbfile: coordinates for the CA trace in PDB format
        • fasta: fasta file with the sequence aligned using map_align
        • emd: EM density map file in nc format
        • level: threshold to apply to the EM density map
        • mrc: EM density file in MRC format for modeller
        • resolution: resolution of the EM map for modeller
        • alpha: residue slices with alpha helix restraints
        • strand: residue slices with beta strand restraints
        • basename: basename of the output modeller files
                    (default: modeller_out)
        """
        self.mrc = mrc
        self.resolution = resolution

        if basename is None:
            self.basename = "modeller_out"
        else:
            self.basename = basename

        self.fasta = fasta
        self.pdbfile = pdbfile
        self.alpha = alpha
        self.strand = strand
        atom_types, aa_list, chain_ids, resids, coords, is_ca_trace = read_pdb(pdbfile)
        if is_ca_trace:
            self.atom_types, self.aa_list, self.chain_ids, self.resids, self.coords, _ = self.ca_to_all()
            self.cmap = get_cmap('%s.pdb'%self.basename) # Contact map for the all atoms protein
            self.n_atoms = self.coords.shape[0]
            self.topology = self.get_topology() # topology (adjacency matrix) for
                                                # the all atom protein
        else:
            # already all atom
            self.atom_types, self.aa_list, self.chain_ids, self.resids, self.coords = \
            atom_types, aa_list, chain_ids, resids, coords
            self.cmap = get_cmap(pdbfile)
        self.emd = emd
        self.level = level

    def ca_to_all(self):
        """
        Convert the CA trace to all atoms
        """
        repair_protein.repair_protein(self.pdbfile, sequence=self.fasta,
                                      write_psf=True,
                                      alpha=self.alpha, strand=self.strand,
                                      outpdb="%s.pdb"%self.basename)
        # return the coordinates of the all atoms protein
        return read_pdb('%s.pdb'%self.basename)

    @property
    def ca_trace(self):
        """
        Coordinates of the CA trace
        """
        selection = numpy.asarray(self.atom_types) == 'CA'
        return self.coords[selection]

    def get_topology(self):
        """
        Read the BONDS section of the PSF to build the corresponding adjacency
        matrix
        """
        psffilename = '%s.psf'%self.basename
        start = False
        topology = numpy.ones((self.n_atoms, self.n_atoms)) * numpy.inf
        with open(psffilename) as psffile:
            for line in psffile:
                if re.findall('!NTHETA', line):
                    start = False
                if start:
                    bonds = numpy.int_(line.split()) - 1
                    for i, j in zip(bonds, bonds[1:]):
                        topology[i, j] = 1
                        topology[j, i] = 1
                if re.findall('!NBOND', line):
                    start = True
        return topology

    def fit(self):
        """
        Fit the all atom structures onto the EM density map
        """
        graph = Graph.Graph(self.topology)
        graph.minimum_spanning_tree = self.topology
        emf = em_fit.Fit(self.emd, self.level, graph, self.coords, 20000,
                         alpha_0=.1, alpha_1=0.0, radius_0=1., radius_1=0.,
                         refine=True)
        emf.learn()
        self.write_pdb(pdbname="%s.pdb"%self.basename)
        repair_protein.repair_protein('%s.pdb'%self.basename,
                                      outpdb=self.basename)
        self.atom_types, self.aa_list, self.chain_ids, self.resids, self.coords, _ = read_pdb('%s.pdb'%self.basename)

    def write_pdb(self, pdbname='modeller_out.pdb'):
        """
        Write a PDB file to disk
        """
        connect = numpy.asarray(numpy.where(~numpy.isinf(self.topology))).T+1
        optimizer.write_pdb(self.coords, sequence=self.aa_list,
                            atoms=self.atom_types, chain_ids=self.chain_ids,
                            resids=self.resids, outfilename=pdbname,
                            write_conect=False)
