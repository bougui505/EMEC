#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-09-11 09:52:06 (UTC+0200)

import os
import sys
import Bio.PDB
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.SCOPData import protein_letters_3to1 as aa3to1
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist


def get_pdb_sequence(structure):
    """
    Return the sequence of the given structure object
    """
    _aainfo = lambda r: (r.id[1], aa3to1.get(r.resname, 'X'))
    seq = [_aainfo(r) for r in structure.get_residues() if is_aa(r)]
    return seq


class PDBalign(object):
    def __init__(self, refpdb, samplepdb):
        self.samplepdb = samplepdb
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)
        self.ref_structure = pdb_parser.get_structure("reference", refpdb)
        self.sample_structure = pdb_parser.get_structure("sample", samplepdb)
        # Use the first model in the pdb-files for alignment
        self.ref_structure = self.ref_structure[0]
        self.ref_sequence = get_pdb_sequence(self.ref_structure)
        self.sample_structure = self.sample_structure[0]
        self.sample_sequence = get_pdb_sequence(self.sample_structure)
        self.res_map = self.align_sequence()
        self.ref_atoms = []
        self.sample_atoms = []
        # Assertion: Only one chain in the structures!
        assert len(self.ref_structure.child_dict.keys()) == 1
        assert len(self.sample_structure.child_dict.keys()) == 1
        ref_chain_id = list(self.ref_structure.child_dict.keys())[0]
        sample_chain_id = list(self.sample_structure.child_dict.keys())[0]
        for ref_res in self.res_map:
            self.ref_atoms.append(self.ref_structure[ref_chain_id][ref_res]['CA'])
            self.sample_atoms.append(self.sample_structure[sample_chain_id][self.res_map[ref_res]]['CA'])

    def align_sequence(self):
        sample_seq = ''.join([i[1] for i in self.sample_sequence])
        ref_seq = ''.join([i[1] for i in self.ref_sequence])
        alns = pairwise2.align.globaldx(sample_seq, ref_seq, matlist.blosum62)
        best_aln = alns[0]
        aligned_A, aligned_B, score, begin, end = best_aln
        mapping = {}
        aa_i_A, aa_i_B = 0, 0
        for aln_i, (aa_aln_A, aa_aln_B) in enumerate(zip(aligned_A, aligned_B)):
            if aa_aln_A == '-':
                if aa_aln_B != '-':
                    aa_i_B += 1
            elif aa_aln_B == '-':
                if aa_aln_A != '-':
                    aa_i_A += 1
            else:
                assert self.sample_sequence[aa_i_A][1] == aa_aln_A
                assert self.ref_sequence[aa_i_B][1] == aa_aln_B
                mapping[self.ref_sequence[aa_i_B][0]] = self.sample_sequence[aa_i_A][0]
                aa_i_A += 1
                aa_i_B += 1
        return mapping

    def align(self):
        super_imposer = Bio.PDB.Superimposer()
        super_imposer.set_atoms(self.ref_atoms, self.sample_atoms)
        super_imposer.apply(self.sample_structure.get_atoms())
        print("REMARK   6 RMSD = %.4f" % super_imposer.rms)

    def write_pdb(self, outfilename=None):
        io = Bio.PDB.PDBIO()
        io.set_structure(self.sample_structure)
        basename = os.path.splitext(self.samplepdb)[0]
        if outfilename is None:
            outfilename = '%s_aligned.pdb' % basename
        io.save(outfilename)


if __name__ == '__main__':
    samplepdb = sys.argv[1]
    refpdb = sys.argv[2]
    try:
        # Check if an output file name is given ...
        outfilename = sys.argv[3]
    except IndexError:
        # ... if not, write the result in std output.
        outfilename = sys.stdout
    pdbalign = PDBalign(refpdb, samplepdb)
    pdbalign.align()
    pdbalign.write_pdb(outfilename=sys.stdout)
