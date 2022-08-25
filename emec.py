#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-05-15 09:59:46 (UTC+0200)

import configparser
import results_parser
import sys
import re
import numpy
import time
import operator
from tempfile import NamedTemporaryFile
import logging
import mrcfile
import EMDensity
import skeleton
import map_align
import optimizer
import all_atoms
import hashlib
import pdbalign
import glob


class Config(object):
    """
    Read the configuration file
    """
    def __init__(self, configfilename):
        """
        • Configuration file name
        """
        self.config = configparser.ConfigParser()
        self.config.read(configfilename)
        # Log data:
        logging.info("EM netCDF filename: %s" % self.nc)
        logging.info("EM level: %.2g" % self.level)
        logging.info("EM resolution: %.2g" % self.resolution)
        logging.info("Pruning threshold: %d" % self.pruning_threshold)
        logging.info("Chain ids: %s" % ' '.join(sorted(self.sequences.keys())))
        for chainid in sorted(self.contacts.keys()):
            logging.info("Chain %s contact file: %s" % (chainid, self.contacts[chainid]))
            logging.info("Chain %s sequence file: %s" % (chainid, self.sequences[chainid]))
        logging.info("Maximum number of iterations for fragment merging: %s" % self.max_iter)
        logging.info("Neighbor distance threshold (in A) for fragment merging: %.2g" % self.neighbor_threshold)

    @property
    def level(self):
        """
        Return the level to apply to the EM density
        """
        return float(self.config.get('EM', 'level'))

    @property
    def threshold(self):
        """
        Distance threshold to compute the contact maps
        """
        return float(self.config.get('Map align', 'threshold'))

    @property
    def nc(self):
        """
        netCDF file name for the EM density
        """
        return self.config.get('EM', 'nc')

    @property
    def resolution(self):
        """
        Read the resolution to use for the MRC file in Modeller refinement
        """
        return float(self.config.get('EM', 'resolution'))

    @property
    def pruning_threshold(self):
        """
        Read the pruning_threshold: minimal size of fragment (in number of CA)
        to keep during the pruning procedure.
        """
        return int(self.config.get('EM', 'pruning_threshold'))

    @property
    def fragments(self):
        """
        Read fragments from the given pdb files instead of computing them
        """
        try:
            globname = self.config.get('Fragments', 'pdbs')
            pdbs = glob.glob(globname)
            coords = []
            for pdb in pdbs:
                _, _, _, _, coords_, _ = all_atoms.read_pdb(pdb)
                coords.append(coords_)
            return coords
        except configparser.NoSectionError:
            return None

    @property
    def max_iter(self):
        """
        Read the maximum number of iterations for the MapAlign based merging of
        fragments
        """
        val = int(self.config.get('Map align', 'max_iter'))
        if val == -1:
            val = numpy.inf
        return val

    @property
    def neighbor_threshold(self):
        """
        Distance threshold (in Angstrom) to define neighboring fragments
        for fragment merging
        """
        val = float(self.config.get('Map align', 'neighbor_threshold'))
        return val

    @property
    def gap_o(self):
        """
        gap opening penalty for the map_align algorithm
        """
        try:
            val = float(self.config.get('Map align', 'gap_o'))
        except configparser.NoOptionError:
            val = -3.
        return val

    @property
    def gap_e(self):
        """
        gap extension penalty for the map_align algorithm
        """
        try:
            val = float(self.config.get('Map align', 'gap_e'))
        except configparser.NoOptionError:
            val = -0.1
        return val

    @property
    def contacts(self):
        """
        Return the dictionary of predicted contacts for each chain.
        self.contacts['A'] gives the filename of the predicted contacts
        for chain A
        """
        chain_dict = {}
        for s in self.config.sections():
            if re.findall('Chain ', s):
                chainid = re.split('Chain ', s)[1]
                chain_dict[chainid] = self.config.get(s, 'contacts')
        return chain_dict

    @property
    def sequences(self):
        """
        Return the dictionnary of sequences for each chain.
        self.sequences['A'] gives the filename of the FASTA file with
        the sequence of chain A
        """
        seq_dict = {}
        for s in self.config.sections():
            if re.findall('Chain ', s):
                chainid = re.split('Chain ', s)[1]
                seq_dict[chainid] = self.config.get(s, 'sequence')
        return seq_dict


def gethash(arr):
    """
    Hash a numpy array
    """
    return hashlib.md5(arr).hexdigest()


class Chains(object):
    """
    A class to store the alignment for each chain
    """
    def __init__(self, config, catraces):
        """
        • config: the config object
        • catraces: the coordinates of the C-alpha traces (numpy array)
        """
        self.chainids = config.contacts.keys()
        self.config = config
        #self.chainids.sort()
        # Get all the C-alpha traces
        self.catraces = catraces
        # Get unique identifier for each fragment
        self.hashes = {gethash(_): i for i, _ in enumerate(catraces)}
        self.maligns = {}  # Dictionnary of map alignment for each chain
        self.aln_score()
        # Alignment scores
        self.scores = {k: self.maligns[k].aln.score for k in self.maligns}
        # Sort the chains by descending score order
        self.chainids = [e[0] for e in\
                              sorted(self.scores.items(),\
                                     key=operator.itemgetter(1))[::-1]]
        self.get_chains()

    def aln_score(self):
        """
        Align all peptides for each chain in alphabetical order for chain ids.
        This method is used to get the global alignment and sort the chain by
        decreasing alignment score value.
        """
        self.maligns = {}
        for chainid in self.chainids:
            print("Global map alignment for chain %s" % chainid)
            malign = map_align.MapAlign(self.catraces,
                                        self.config.nc,
                                        self.config.level,
                                        self.config.contacts[chainid],
                                        self.config.sequences[chainid],
                                        neighbor_threshold=self.config.neighbor_threshold,
                                        gap_o=self.config.gap_o,
                                        gap_e=self.config.gap_e,
                                        max_iter=self.config.max_iter,
                                        threshold=self.config.threshold)
            malign.align_fragments()
            self.maligns[chainid] = malign

    def get_chains(self):
        """
        Get the final chain assignment
        """
        chainid = self.chainids[0]
        print("Build CA trace for chain %s" % chainid)
        # List of unique identifiers for catraces
        fragments_used = [gethash(self.catraces[i])\
                          for i in self.maligns[chainid].adjmat.max_score().key[0]]
        self.maligns[chainid].fragments_hash = fragments_used
        info = [self.hashes[_] for _ in fragments_used]
        logging.info("Fragments attributed to chain %s: %s" % (chainid, info))
        for chainid in self.chainids[1:]:
            print("Build CA trace for chain %s" % chainid)
            catraces = [catrace for catrace in self.catraces\
                        if gethash(catrace)\
                        not in fragments_used]
            print("Remaining number of fragments to merge %d" % len(catraces))
            malign = map_align.MapAlign(catraces,
                                        self.config.nc,
                                        self.config.level,
                                        self.config.contacts[chainid],
                                        self.config.sequences[chainid],
                                        neighbor_threshold=self.config.neighbor_threshold,
                                        gap_o=self.config.gap_o,
                                        gap_e=self.config.gap_e,
                                        max_iter=self.config.max_iter,
                                        threshold=self.config.threshold)
            malign.align_fragments()
            fragments_hash = [gethash(catraces[i])\
                              for i in malign.adjmat.max_score().key[0]]
            malign.fragments_hash = fragments_hash
            fragments_used.extend(fragments_hash)
            info = [self.hashes[_] for _ in fragments_hash]
            logging.info("Fragments attributed to chain %s: %s" % (chainid, info))
            self.maligns[chainid] = malign

    def fix_chains(self):
        """
        Fix the chain crosslinks based on the map alignment
        attribution.
        The class split the chain into new fragments that can be realigned.
        """
        fragment_used = []  # Set of fragments attributed to chains
        all_fragments = set(range(len(self.catraces)))
        new_fragments = []  # List containing the coordinates of new fragments
        for chainid in self.chainids:
            seq = self.maligns[chainid].aln.sequence
            cutting_spot = list((numpy.asarray(zip(seq, seq[1:])) ==\
                                ['X', 'X']).sum(axis=1) == 1)
            cutting_spot.append(False)
            cutting_spot = numpy.where(cutting_spot)[0]
            new_fragments.extend(numpy.split(self.maligns[chainid].adjmat.max_score().coords, cutting_spot))
            fragment_used.extend([self.hashes[_] for _ in self.maligns[chainid].fragments_hash])
        remaining_fragments = list(all_fragments - set(fragment_used))
        remaining_fragments = list(numpy.asarray(self.catraces)[remaining_fragments])
        new_fragments.extend(remaining_fragments)
        new_fragments = [_ for _ in new_fragments if len(_) > 1]
        return new_fragments


def fetch_modeller_objective_function(pdbfilename):
    """
    Fetch the modeller objective function value
    from the header of the given pdbfilename
    """
    with open(pdbfilename, 'r') as pdbfile:
        for line in pdbfile:
            if re.match("(.*)MODELLER OBJECTIVE FUNCTION(.*)", line):
                string = line[11:-1]
    return string


class CAtoAll(object):
    """
    Convert Calpha trace to all atom with sequence attribution
    """
    def __init__(self, config, chains):
        """
        • config: the config object
        • chains: chains object containing all the chains
        """
        self.config = config
        self.chains = chains
        # List of chain ids (e.g. ['A', 'C', 'B']):
        self.chainids = self.chains.chainids
        print("Building atomic models for chains: %s" % self.chainids)
        self.get_all_atoms()

    def get_all_atoms(self):
        """
        Get the all atoms models for each chain in parallel
        """
        for chainid in self.chainids:
            print("Building all atom model for chain %s" % chainid)
            data = self.refine(chainid)

    def refine(self, chainid):
        """
        Optimize the alignment and get the all atom models.
        • chainid: Id of the chain to refine
        """
        em_level = -numpy.inf
        malign = self.chains.maligns[chainid]
        malign.aln = map_align.map_align(malign.coords, malign.gmap, fasta_file=malign.fasta_file)
        modeller_out_basename = "model_%s" % (chainid)
        aa = all_atoms.AllAtoms('chain_%s.pdb' % chainid,
                                self.config.nc,
                                em_level,
                                "chain_%s.mrc" % chainid,
                                self.config.resolution,
                                basename=modeller_out_basename)
        logging.info("Modeller: Chain %s: %s" %
                     (chainid, fetch_modeller_objective_function("%s.pdb" % modeller_out_basename)))
        malign.coords_best = aa.ca_trace
        # map alignment after modeller optimization for all atoms
        malign.aln = map_align.map_align(aa.cmap, malign.gmap, fasta_file=malign.fasta_file)
        logging.info("Modeller: Chain %s: contact map alignment score: %.4g" % (chainid, malign.aln.score))
        logging.info("Modeller: Chain %s: number of residues aligned: %d/%d" %
                     (chainid, malign.aln.n_align, malign.aln.npos))
        coverage = float(malign.aln.n_align) / malign.aln.npos
        logging.info("Modeller: Chain %s: alignment coverage: %.4g" % (chainid, coverage))
        # Align the PDB to the initial threading:
        aligner = pdbalign.PDBalign('chain_%s.pdb' % chainid, '%s.pdb' % modeller_out_basename)
        aligner.align()
        aligner.write_pdb(outfilename='%s.pdb' % modeller_out_basename)
        return malign


def zone(ncfile, coords, outmrcfilename, distance_threshold=4.):
    """
    Zone selection of a EM density around the given coordinates
    """
    emd = EMDensity.Density(ncfile, -numpy.inf)
    zone = []
    [zone.extend(list(e)) for e in emd.kdtree.query_ball_point(coords, distance_threshold)]
    zone = numpy.asarray(zone)
    filter_set = set(range(len(emd.density))) - set(zone)
    emd.density[list(filter_set)] = 0.
    data = emd.density.reshape(emd.nx, emd.ny, emd.nz).T
    # Zero filling to have a cubic map for modeller
    nx, ny, nz = data.shape
    maxdim = max(nx, ny, nz)
    data = numpy.pad(data, ((0, maxdim - nx), (0, maxdim - ny), (0, maxdim - nz)), 'constant', constant_values=(0, 0))
    mrc = mrcfile.new(outmrcfilename, data=data, overwrite=True)
    mrc.update_header_from_data()
    mrc.voxel_size = (emd.x_step, emd.y_step, emd.z_step)
    mrc.header.origin = (emd.xgrid[0], emd.ygrid[0], emd.zgrid[0])
    mrc.close()


if __name__ == '__main__':
    logging.basicConfig(filename='emec.log', format='%(levelname)s:%(message)s', level=logging.DEBUG)
    config = Config(sys.argv[1])
    print(" Compute the Graph from the EM density map...")
    nc, level = config.nc, config.level
    if config.fragments is None:
        skl = skeleton.Skeleton(nc, level, config.pruning_threshold)
        for i, chain in enumerate(skl.chains):
            optimizer.write_pdb(chain.coords_CA, outfilename='fragment_%d.pdb' % i)
        # Get all the C-alpha traces
        catraces = [chain.coords_CA for chain in skl.chains]
    else:
        catraces = config.fragments
    chains = Chains(config, catraces)
    # Write the C-alpha trace for each chain as a PDB format
    # and the corresponding segmented EM map.
    traces = results_parser.Traces(chains)
    for chainid in chains.chainids:
        peptides = traces.get_traces(chainid)
        if len(peptides) > 0:
            traces.write_ca_trace(peptides[0], 'chain_%s.pdb' % chainid)
    all_atoms = CAtoAll(config, chains)
