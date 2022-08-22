#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-05-15 09:59:46 (UTC+0200)

import configparser
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

from mpi4py import MPI


def barrier(comm, tag=0, sleep=0.01):
    """
    MPI barrier fonction
    that solve the problem that Idle process occupies 100% CPU.
    See: https://goo.gl/NofOO9
    """
    size = comm.Get_size()
    if size == 1:
        return
    rank = comm.Get_rank()
    mask = 1
    while mask < size:
        dst = (rank + mask) % size
        src = (rank - mask + size) % size
        req = comm.isend(None, dst, tag)
        while not comm.Iprobe(src, tag):
            time.sleep(sleep)
        comm.recv(None, src, tag)
        req.Wait()
        mask <<= 1


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
        barrier(COMM)
        if RANK == 0:
            for chainid in sorted(self.chainids):
                malign = self.maligns[chainid]
                logging.info("Chain %s: contact map alignment score: %.4g" % (chainid, malign.aln.score))
                logging.info("Chain %s: number of residues aligned: %d/%d" %
                             (chainid, malign.aln.n_align, malign.aln.npos))
                coverage = float(malign.aln.n_align) / malign.aln.npos
                logging.info("Chain %s: alignment coverage: %.4g" % (chainid, coverage))

    def aln_score(self):
        """
        Align all peptides for each chain in alphabetical order for chain ids.
        This method is used to get the global alignment and sort the chain by
        decreasing alignment score value.
        """
        barrier(COMM)
        if RANK == 0:
            njobs = len(self.chainids)
            k = int(numpy.ceil(float(njobs) / SIZE))
            job_array = range(njobs)
            job_array.extend([
                None,
            ] * (k * SIZE - njobs))  # Add None values if not a multiple of SIZE
            job_array = numpy.asarray(job_array).reshape(k, SIZE)
        else:
            job_array = None
        job_array = COMM.bcast(job_array, root=0)
        harvest = []  # to store the results of the gathering [(job_id1: result1), ()job_id2: result2, ...]
        for job_ids in job_array:
            job_id = COMM.scatter(job_ids, root=0)
            if job_id is not None:
                chainid = self.chainids[job_id]
                print("Global map alignment for chain %s" % chainid)
                malign = map_align.MapAlign(self.catraces,
                                            self.config.nc,
                                            self.config.level,
                                            self.config.contacts[chainid],
                                            self.config.sequences[chainid],
                                            neighbor_threshold=self.config.neighbor_threshold)
                malign.align_fragments(max_iter=self.config.max_iter)
                data = (chainid, malign)
            gather = COMM.gather(data, root=0)
            if gather is not None:
                harvest.extend(gather)
            barrier(COMM)
        barrier(COMM)
        if RANK == 0:
            self.maligns = dict(harvest)

    def get_chains(self):
        """
        Get the final chain assignment
        """
        barrier(COMM)
        if RANK == 0:
            chainid = self.chainids[0]
            print("Build CA trace for chain %s" % chainid)
            # List of unique identifiers for catraces
            fragments_used = [gethash(self.catraces[i])\
                              for i in self.maligns[chainid].adjmat.max_score().key[0]]
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
                                            neighbor_threshold=self.config.neighbor_threshold)
                malign.align_fragments(max_iter=self.config.max_iter)
                fragments_hash = [gethash(catraces[i])\
                                  for i in malign.adjmat.max_score().key[0]]
                fragments_used.extend(fragments_hash)
                info = [self.hashes[_] for _ in fragments_hash]
                logging.info("Fragments attributed to chain %s: %s" % (chainid, info))
                self.maligns[chainid] = malign


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
        barrier(COMM)
        if RANK == 0:
            njobs = len(self.chainids)
            k = int(numpy.ceil(float(njobs) / SIZE))
            job_array = range(njobs)
            job_array.extend([
                None,
            ] * (k * SIZE - njobs))  # Add None values if not a multiple of SIZE
            job_array = numpy.asarray(job_array).reshape(k, SIZE)
        else:
            job_array = None
        job_array = COMM.bcast(job_array, root=0)
        harvest = []  # to store the results of the gathering [(job_id1: result1), ()job_id2: result2, ...]
        for job_ids in job_array:
            job_id = COMM.scatter(job_ids, root=0)
            if job_id is not None:
                chainid = self.chainids[job_id]
                print("Job #%d: Building all atom model for chain %s" % (job_id, chainid))
                data = self.refine(chainid)
            gather = COMM.gather(data, root=0)
            if gather is not None:
                harvest.extend(gather)
            barrier(COMM)

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
                                self.config.sequences[chainid],
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
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()  # Number of CPUS
    RANK = COMM.Get_rank()

    logging.basicConfig(filename='emec.log', format='%(levelname)s:%(message)s', level=logging.DEBUG)
    barrier(COMM)
    if RANK == 0:
        config = Config(sys.argv[1])
        print(" Compute the Graph from the EM density map...")
        nc, level = config.nc, config.level
        skl = skeleton.Skeleton(nc, level, config.pruning_threshold)
        for i, chain in enumerate(skl.chains):
            optimizer.write_pdb(chain.coords_CA, outfilename='fragment_%d.pdb' % i)
        # Get all the C-alpha traces
        catraces = [chain.coords_CA for chain in skl.chains]
    else:
        config = None
        catraces = None
    barrier(COMM)
    config = COMM.bcast(config, root=0)
    catraces = COMM.bcast(catraces, root=0)
    chains = Chains(config, catraces)
    chains = COMM.bcast(chains, root=0)
    barrier(COMM)
    if RANK == 0:
        # Write the C-alpha trace for each chain as a PDB format
        # and the corresponding segmented EM map.
        for chainid in chains.chainids:
            optimizer.write_pdb(chains.maligns[chainid].adjmat.max_score().coords,
                                outfilename='chain_%s.pdb' % chainid,
                                sequence=chains.maligns[chainid].aln.sequence)
            zone(config.nc, chains.maligns[chainid].adjmat.max_score().coords, outmrcfilename='chain_%s.mrc' % chainid)
    barrier(COMM)
    all_atoms = CAtoAll(config, chains)
    barrier(COMM)
