#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-03-16 10:17:18 (UTC+0100)

import all_atoms
import EMDensity
from subprocess import Popen
from subprocess import PIPE
import numpy
import re

def list_to_slices(inputlist):
    """
    Convert a flatten list to a list of slices:
    test = [0,2,3,4,5,6,12,99,100,101,102,13,14,18,19,20,25]
    list_to_slices(test)
    -> [(0, 0), (2, 6), (12, 14), (18, 20), (25, 25), (99, 102)]
    """
    inputlist.sort()
    pointers = numpy.where(numpy.diff(inputlist) > 1)[0]
    pointers = zip(numpy.r_[0, pointers+1], numpy.r_[pointers, len(inputlist)-1])
    slices = [(inputlist[i], inputlist[j]) for i, j in pointers]
    return slices

class DSSP(object):
    """
    Python class for the DSSP program for secondary structure attribution from a
    PDB protein structure file.
    """
    def __init__(self, pdbfilename, emd, level):
        """
        • pdbfilename: name of the pdb file
        • emd: Electron microscopy map file in nc format
        • level: thresholding level to apply on the EM map
        """
        self.pdbfilename = pdbfilename
        self.emd = EMDensity.Density(emd, level)
        # Secondary Structure attribution of the EM density map (internal use)
        self._emdss = numpy.chararray(self.emd.density.shape)
        self._emdss[:] = '-'
        self._stdoutdata, self._stderrdata = None, None
        self.atom_types, self.aa_list, self.chain_ids, self.resids,\
        self.coords = None, None, None, None, None

    @property
    def ss_list(self):
        """
        Read and format the DSSP output
        Returns ss_list, the list of secondary structure attribution for each
        amino acid residue of the given pdb file.
        H   Alpha helix
        B   Beta bridge
        E   Strand
        G   Helix-3
        I   Helix-5
        T   Turn
        S   Bend
        """
        sbp = Popen(['dssp', '-i', self.pdbfilename], stdout=PIPE, stderr=PIPE)
        self._stdoutdata, self._stderrdata = sbp.communicate()
        start = False
        ss_list = []
        for line in self._stdoutdata.split('\n'):
            if start:
                try:
                    if line[16] == " ":
                        ss = "-"
                    else:
                        ss = line[16]
                    ss_list.append(ss)
                except IndexError:
                    break
            if re.findall("#  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O    O-->H-N    N-H-->O    O-->H-N    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA",
                          line):
                start = True
        return ss_list

    @property
    def ca_trace(self):
        """
        CA trace coordinates
        """
        self.atom_types, self.aa_list, self.chain_ids, self.resids,\
        self.coords, _ = all_atoms.read_pdb(self.pdbfilename)
        return self.coords[numpy.asarray(self.atom_types) == 'CA']

    @property
    def emdss(self):
        """
        • Secondary Structure attribution of the EM density map:
            ‣ H: Alpha helix
            ‣ E: Strand
            ‣ -: Coil
        """
        ca_trace = self.ca_trace
        ss_list = self.ss_list
        for i, sstype in enumerate(ss_list):
            if sstype in ['H', 'E']:
                coord = ca_trace[i]
                # Pointers in the EM density for the neighborhood of the CA coordinates
                pointers = self.emd.kdtree.query_ball_point(coord, 3.8)
                self._emdss[pointers] = sstype
        return self._emdss

    def get_ss_restraints(self, pdbfilename):
        """
        Get the secondary structure restraints for the given coordinates of the
        CA-trace.
        • pdbfilename: Name of the PDB structure file to compute the restraints on
        Returns:
        • (alpha, strand): the list of resids in alpha helix and beta strands
        respectively
        """
        _, _, _, _, coords, is_ca_trace = all_atoms.read_pdb(pdbfilename)
        if is_ca_trace:
            alpha = []
            strand = []
            emdss = self.emdss
            for i, coord in enumerate(coords):
                pointer = self.emd.kdtree.query(coord)[1]
                sstype = emdss[pointer]
                if sstype == 'H':
                    # 1-based numbering for resid selection...
                    alpha.append(i+1)
                elif sstype == 'E':
                    # 1-based numbering for resid selection...
                    strand.append(i+1)
            if len(alpha) > 0:
                alpha = list_to_slices(alpha)
            else:
                alpha = None
            if len(strand) > 0:
                strand = list_to_slices(strand)
            else:
                strand = None
            return alpha, strand
        else:
            print "%s is not a valid pdbfile for a CA-trace structure"%pdbfilename
            return None, None
