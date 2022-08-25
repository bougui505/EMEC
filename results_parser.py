#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-12-15 11:02:38 (UTC+0100)
"""
Simple script to read chains.pkl file and parse the results
"""

from mast import Chains
from mast import Config
from mast import zone
import pickle
import numpy
import os
import optimizer
import all_atoms
import pdbalign


class Traces(object):
    """
    Parse possible protein topologies from chains
    """
    def __init__(self, chains):
        """
        • chains: object created by main mast script
        """
        self.chains = chains
        self.chain_ids = list(self.chains.maligns.keys())  # list of chainids (e.g. ['A', 'B', 'C'])
        self.chain_ids.sort()

    def get_traces(self, chain_id):
        """
        Get the different traces for the given chain_id, sorted by
        decreasing score (the higher, the better)
        """
        malign = self.chains.maligns[chain_id]  # Map align object
        peptides = []
        for key in malign.adjmat.scores:
            if malign.adjmat.scores[key] != -numpy.inf:
                peptide = malign.adjmat.get(key)
                peptide.aln.sequence = malign.sequence  # Input sequence (string)
                peptide.aln.get_seq()  # Get the aligned sequence (list)
                peptides.append(peptide)
        scores = [_.score for _ in peptides]
        scores = numpy.asarray(scores)
        peptides = numpy.asarray(peptides)
        sorter = scores.argsort()[::-1]
        peptides = peptides[sorter]
        return peptides

    def write_ca_trace(self, peptide, outfilename):
        """
        Write the Calpha trace to a pdb file for the given peptide
        """
        # Get correct numbering for residue ids based on fasta sequence
        resids = numpy.asarray(list(peptide.aln.alignment.values())) + 1
        optimizer.write_pdb(peptide.coords,
                            resids=resids,
                            outfilename=outfilename,
                            sequence=peptide.aln.sequence,
                            write_UNK=False)
        # Write the corresponding segmented map in mrc file
        zone(self.chains.config.nc, peptide.coords, outmrcfilename='%s.mrc' % os.path.splitext(outfilename)[0])

    def get_model(self, peptide, outfilename):
        """
        Get the full atom model for the given peptide
        """
        self.write_ca_trace(peptide, outfilename)
        mrcfilename = '%s.mrc' % os.path.splitext(outfilename)[0]
        modeller_out_basename = 'model_%s' % os.path.splitext(outfilename)[0]
        aa = all_atoms.AllAtoms(outfilename,
                                self.chains.config.nc,
                                self.chains.config.level,
                                mrcfilename,
                                self.chains.config.resolution,
                                basename=modeller_out_basename)
        # Align the PDB to the initial threading:
        aligner = pdbalign.PDBalign(outfilename, '%s.pdb' % modeller_out_basename)
        aligner.align()
        aligner.write_pdb(outfilename='%s.pdb' % modeller_out_basename)


if __name__ == '__main__':
    with open('chains.pkl') as infile:
        chains = pickle.load(infile)
    traces = Traces(chains)
