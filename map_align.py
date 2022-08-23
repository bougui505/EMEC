#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-02-07 10:03:29 (UTC+0100)

import numpy
import scipy.spatial.distance
import scipy.spatial
from subprocess import Popen
from subprocess import PIPE
from tempfile import NamedTemporaryFile
import progress_reporting
import EMDensity
import Graph
import em_fit
import scipy.cluster.hierarchy
import operator
import copy


class Alignment(object):
    """
    Alignment object that contain alignment informations
    """
    def __init__(self, alignment, score, n_align, fasta_file=None, npos=None):
        """
        • alignment: Dictionnary of the alignment:
        {resid_input: resid_target, ...}
        • score: score of the alignment
        • n_align: number of residues aligned
        • fasta_file: fasta file with the sequence (optional)
        • npos: The total number of AA (position) in the sequence
        • outfilename: name of the fasta file containing the final alignment
        """
        self.alignment = alignment
        self.score = score
        self.n_align = n_align
        self.npos = npos
        if fasta_file is not None:
            self.sequence = read_sequence(fasta_file)
            self.get_seq()
        else:
            self.sequence = None

    def get_seq(self):
        """
        Get the aligned sequence.
        """
        seq = []  # Sequence aligned
        for i in range(self.npos):  # Iterate over the C alpha position
            if self.alignment.has_key(i):
                # The bead is aligned with the contact map
                seq.append(self.sequence[self.alignment[i]])
            else:
                # The bead is not aligned: no sidechain, put a glycine
                seq.append('G')
        self.sequence = seq


class Adjmat(object):
    """
    Class to store the adjacency matrix for peptide object
    """
    def __init__(self):
        self.peptides = {}

    def append(self, peptide):
        """
        Append peptide object to the adjmat
        """
        self.peptides[peptide.key] = peptide

    @property
    def scores(self):
        """
        Get the list of alignment scores for peptides
        """
        return {p.key: p.score for p in self.peptides.values()}

    @property
    def keys(self):
        """
        Return the peptide keys contained in adjmat object
        """
        return self.peptides.keys()

    def get(self, key):
        """
        Return the peptide object contained in adjmat with the corresponding key
        """
        return self.peptides[key]

    def max_score(self, exclusion_set=[]):
        """
        Get the peptide with the maximal alignment score
        • exclusion_set: peptide ids to exclude from the max_score search
        """
        sorted_peptides = sorted(self.scores.items(), key=operator.itemgetter(1), reverse=True)
        keys = [p[0] for p in sorted_peptides\
                    if p[0] not in exclusion_set]
        return self.peptides[keys[0]]


def get_cmap(coords, threshold=8.):
    """
    compute the contact map from the coordinates (coords)
    • coords: numpy array (n*3)
    • threshold: distance threshold to define a contact
    """
    cmap = scipy.spatial.distance.pdist(coords)
    cmap = scipy.spatial.distance.squareform(cmap)
    cmap = numpy.int_(cmap <= threshold)
    return cmap


def write_cmap(cmap, outfilename=None):
    """
    Write contact map to ascci file that complies with map_align cpp binary
    """
    indices = numpy.asarray(numpy.where(numpy.triu(cmap))).T
    n = cmap.shape[0]
    if outfilename is None:
        # Create a temporary file
        outfile = NamedTemporaryFile('w', delete=True)
    else:
        outfile = open(outfilename, 'w')
    outfile.write('LEN %d\n' % n)
    for e in indices:
        outfile.write('CON %d %d 1.0\n' % tuple(e))
    return outfile


def read_gremlin(gremlin_file):
    """
    Read the gremlin formatted output file
    """
    gremlin = numpy.int_(numpy.genfromtxt(gremlin_file, skip_header=1)[:, :2])
    naa = gremlin.max()
    gmap = numpy.zeros((naa, naa))
    for i, j in gremlin:
        gmap[i - 1, j - 1] = 1
        gmap[j - 1, i - 1] = 1
    return gmap


def is_cmap(matrix):
    """
    Check if the given matrix is a contact map
    """
    n, p = matrix.shape
    return numpy.logical_and(n == p, (set(numpy.unique(matrix)) == set([0, 1])))


def map_align(inputdata, cmap_target, fasta_file=None, threshold=8., gap_o=-3., gap_e=0., n_iter=20, outfilename=None):
    """
    Compute the contact map (with the given threshold) from the coordinates
    (inputdata) and align it to the given contact map (cmap_target).
    • inputdata: coordinates or contact map
    • fasta_file: Fasta file containining the sequence
    • gap_o: gap opening penalty
    • gap_e: gap extension penalty
    • n_iter: number of iterations for map_align program
    • outfilename: Name of the output Fasta file name containing the final
                   alignment.
    """
    if is_cmap(inputdata):
        # inputdata is already a contact map
        cmap_input = write_cmap(inputdata)
    else:
        # inputdata are coordinates of CA trace.
        # Compute the corresponding contact map
        cmap_input = write_cmap(get_cmap(inputdata, threshold=threshold))
    cmap_target = write_cmap(cmap_target)
    sbp = Popen([
        'map_align', '-gap_o',
        '%.2f' % gap_o, '-gap_e',
        '%.2f' % gap_e, '-a', cmap_input.name, '-b', cmap_target.name, '-silent', '-iter',
        '%d' % n_iter
    ],
                stdout=PIPE,
                stderr=PIPE)
    stdoutdata, stderrdata = sbp.communicate()
    cmap_input.close()
    cmap_target.close()
    out_splitted = stdoutdata.split()
    if len(out_splitted) > 0:  # the alignment works
        score = float(out_splitted[6])  # alignment score
        n_align = int(out_splitted[7])  # number of residues aligned
        # Correct the alignment score by removing the number of residues not aligned
        score -= inputdata.shape[0] - n_align
        # Dictionnary of the alignment:
        # {resid_input: resid_target, ...}
        alignment = dict([numpy.int_(e.split(':')) for e in out_splitted[8:]])
        return Alignment(alignment, score, n_align, fasta_file=fasta_file, npos=inputdata.shape[0])
    else:
        return None


def read_sequence(fasta_file):
    """
    Read a sequence from a fasta file
    """
    data = numpy.genfromtxt(fasta_file, skip_header=1, dtype=str)
    if numpy.size(data.shape) > 0:  # multiple lines
        seq = ''.join(data)
    else:
        seq = str(data)
    return seq


class Peptide(object):
    """
    Return the CA trace coordinates of the peptide in one direction or the other
    based on the map align score
    """
    def __init__(self, coords, aln, key):
        """
        • coords: coordinates of the peptide
        • aln: alignment object
        • key: identifier for the peptide (i, order), with:
            • i: the index of the peptide
            • order: 1 or -1 for normal or reverse order
        • self.nca: number of ca
        • self.score: score of the given aln
        """
        self.coords = coords
        self.aln = aln
        self.key = key
        self.nca = self.coords.shape[0]
        self.neighbors_r, self.neighbors_l = None, None
        self.components = None

    @property
    def score(self):
        """
        Link to the alignment score
        """
        if self.aln is not None:
            score = self.aln.score
        else:
            score = -numpy.inf
        return score

    @property
    def n_comp(self):
        """
        Return the number of components of the peptide
        """
        if self.components is None:
            return 1
        else:
            return len(self.components)

    @property
    def n_gap(self):
        """
        Number of gap in the alignment
        """
        return (numpy.diff(self.aln.alignment.values()) - 1).sum()

    @property
    def max_gap_size(self):
        """
        Size of the biggest gap
        """
        return (numpy.diff(self.aln.alignment.values()) - 1).max()

    @property
    def gap_penalty(self):
        """
        self.n_gap / self.nca
        """
        return self.n_gap / float(self.nca)


class MapAlign(object):
    """
    Align contact maps based on
    https://github.com/sokrypton/map_align
    """
    def __init__(self,
                 catraces,
                 emd,
                 level,
                 gremlin_file,
                 fasta_file,
                 threshold=8.,
                 neighbor_threshold=11.,
                 n_iter=1.,
                 gap_o=-3.,
                 gap_e=-0.1):
        """
        • catraces: list of coordinates for the C-alpha traces
        list of array of coordinates
        • emd: electron microscopy density file name
        • level: threshold to apply on the electron microscopy density
        • threshold: distance threshold to define a contact
        • gremlin_file: name of the gremlin result file with predicted contacts
          gremlin_file could also be a contact map as a numpy array
        • fasta_file: Fasta file containining the sequence
        • neighbor_threshold: maximal distance to define neighbors of a peptide.
          The neighbors are defined as the peptides whose N-Ter are distant to
          less than neighbor_threshold from the C-Ter of the considered peptide.
        • n_iter: number of iterations for the map_align algorithm
        • gap_o: gap opening penalty for the map_align algorithm
        • gap_e: gap extension penalty for the map_align algorithm
        """
        self.threshold = threshold
        self.neighbor_threshold = neighbor_threshold
        self.n_iter = n_iter
        self.gap_o = gap_o
        self.gap_e = gap_e
        catraces = numpy.asarray(catraces)
        if type(gremlin_file) is numpy.ndarray:
            # gremlin_file is already a contact map
            self.gmap = gremlin_file
        else:
            self.gmap = read_gremlin(gremlin_file)  # gremlin predicted contact map
        self.n_peptide = catraces.shape[0]
        self.peptides = numpy.asarray([Peptide(catrace, None, (i, 1)) for i, catrace in enumerate(catraces)])
        # Compute peptide alignment individually in both order
        self.get_peptide_alignment()
        self.anchors = self.get_anchors()
        self.emd = emd
        self.level = level
        self.coords_best = None  # Coordinates after optimization
        self.aln = None  # Alignment object after optimization
        self.fasta_file = fasta_file
        self.adjmat = Adjmat()  # Adjacency matrix for peptide objects
        # Compute a KDTree on all the coordinates for neighbor search
        pool = self.peptides.values()
        data = numpy.asarray([p.coords[0] for p in pool])
        self.keys = numpy.asarray([p.key for p in pool])
        self.kdtree = scipy.spatial.cKDTree(data)

    def get_anchors(self):
        """
        Define the anchoring peptides as the peptide with the best scores
        The anchors are sorted from the best score to the worst one.
        """
        anchors = []
        for pept_id in range(self.n_peptide):
            pept = []
            for order in [1, -1]:
                pept.append(self.peptides[(pept_id, order)])
            scores = [p.score for p in pept]
            if sum(scores) > 0.:
                anchor = pept[numpy.argmax(scores)]
                anchors.append(anchor)
        # sort anchors:
        scores = [p.score for p in anchors]
        sorter = numpy.argsort(scores)[::-1]
        anchors = numpy.asarray(anchors)[sorter]
        return anchors

    @property
    def anchor_overlaps(self):
        """
        Return the overlap score between the anchors (pairwise matrix).
        The overlap is normalized (1: total overlap, 0: no overlap)
        Two peptides overlap if they aligned in the same region.
        """
        overlap = []
        for i, a1 in enumerate(self.anchors):
            region1 = set(a1.aln.alignment.values())
            for a2 in self.anchors[i + 1:]:
                region2 = set(a2.aln.alignment.values())
                minlen = float(min([len(region1), len(region2)]))
                overlap.append(len(region1 & region2) / minlen)
        return scipy.spatial.distance.squareform(overlap)

    @property
    def coords(self):
        """
        Return the coordinates of all the peptides concatenated
        """
        if self.coords_best is None:  # Before optimization
            return numpy.concatenate([self.peptides[tuple(key)].coords\
                                      for key in self.keys])
        else:
            return self.coords_best

    def get_pdist(self):
        """
        Pairwise distance matrix between peptides.
        The element i, j of this matrix is defined as the minimal distance of
        the pairwise atomic distance between peptide i and j
        """
        pdist = []
        for i in range(self.n_peptide):
            for j in range(i + 1, self.n_peptide):
                pept1, pept2 = self.peptides[(i, 1)], self.peptides[(j, 1)]
                pdist.append(scipy.spatial.distance.cdist(pept1.coords, pept2.coords).min())
        return scipy.spatial.distance.squareform(pdist)

    def get_neighbors(self, peptide):
        """
        Compute the given peptide
        • self.neighbor_threshold: maximal distance to define neighbors of a
          peptide.
          The neighbors are defined as the peptides whose N-Ter are distant to
          less than neighbor_threshold from the C-Ter of the considered peptide.
        """
        if type(peptide.key[0]) is int:
            keys = [
                peptide.key[0],
            ]
        else:
            keys = peptide.key[0]
        # Right neighbors:
        neighbors = self.keys[self.kdtree.query_ball_point(peptide.coords[-1],\
                                                       self.neighbor_threshold)]
        neighbors = [tuple(e) for e in neighbors if e[0] not in keys]
        peptide.neighbors_r = [self.peptides[i] for i in neighbors]
        # Left neighbors:
        neighbors = self.keys[self.kdtree.query_ball_point(peptide.coords[0],\
                                                       self.neighbor_threshold)]
        neighbors = [(k, -o) for (k, o) in neighbors if k not in keys]
        peptide.neighbors_l = [self.peptides[i] for i in neighbors]
        return peptide

    def merge(self, *peptides):
        """
        Merge peptides together and compute the corresponding alignment
        """
        coords = numpy.concatenate([e.coords for e in peptides])
        keys = tuple([e.key[0] for e in peptides])
        keys = tuple(numpy.r_[keys])
        orders = tuple([e.key[1] for e in peptides])
        orders = tuple(numpy.r_[orders])
        concat_key = (keys, orders)
        if concat_key not in self.adjmat.keys:
            aln = map_align(coords, self.gmap, n_iter=1, gap_o=self.gap_o, gap_e=self.gap_e)
            peptide = Peptide(coords, aln, concat_key)
            self._get_components(peptide)
            self._get_coverage(peptide)
            self.adjmat.append(peptide)
            return peptide
        else:
            return self.adjmat.get(concat_key)

    def delete(self, peptide, index):
        """
        Delete the peptide with the given index from the peptide
        """
        if index < 0:
            index = len(peptide.components) + index
        peptides = [p for i, p in enumerate(peptide.components) if i != index]
        return self.merge(*peptides)

    def flip(self, peptide, index):
        """
        Flip the peptide (change the order) with the given index in the peptide
        """
        if index < 0:
            index = len(peptide.components) + index
        keys = [p.key for p in peptide.components]
        peptides = [self.peptides[(k,o)]\
                    if i != index else self.peptides[(k,-o)]\
                    for i, (k,o) in enumerate(keys)]
        return self.merge(*peptides)

    def replace(self, peptide, index, newpeptide):
        """
        Replace the component given by index in the peptide by newpeptide
        """
        if index < 0:
            index = len(peptide.components) + index
        peptides = [p if i != index else newpeptide\
                    for i, p in enumerate(peptide.components)]
        return self.merge(*peptides)

    def swap(self, peptide, index1, index2):
        """
        Swap the components index1 and index2 in peptide
        """
        peptides = [p for p in peptide.components]
        peptides[index1], peptides[index2] = peptides[index2], peptides[index1]
        return self.merge(*peptides)

    def _get_components(self, peptide):
        """
        Split the multi peptide to individual compoenents
        """
        start = 0
        components = []
        if peptide.aln is not None:
            for key in zip(*peptide.key):
                pept_ = copy.deepcopy(self.peptides[key])
                nca = pept_.nca
                alignment = {k-start: v for (k, v) in peptide.aln.alignment.iteritems()\
                             if k in range(start, start+nca)}
                pept_.aln = copy.deepcopy(peptide.aln)
                pept_.aln.alignment = alignment
                components.append(copy.deepcopy(pept_))
                start += nca
            peptide.components = components

    def _get_coverage(self, peptide):
        """
        Compute the alignment coverage
        """
        if peptide.aln is not None:
            coverage = peptide.aln.n_align / float(self.gmap.shape[0])
        else:
            coverage = 0
        peptide.coverage = coverage

    def split(self, peptide, threshold=None):
        """
        Split the peptide in multiple peptides at positions where there is a gap
        greater than threshold.
        • If threshold is None, the threshold is set as the length of the
        smallest peptide.
        """
        if threshold is None:
            threshold = min([p.nca for p in self.peptides.values()])
        if peptide.max_gap_size > threshold and peptide.n_comp > 2:
            # Intra peptide gaps
            intragaps = [p.max_gap_size for p in peptide.components]
            limits = numpy.asarray([numpy.asarray(p.aln.alignment.values())[[0,-1]]\
                                    for p in peptide.components])
            intergaps = []
            for i in range(limits.shape[0] - 1):
                intergaps.append(limits[i + 1][0] - limits[i][-1])
            # Assign the maximum of intragaps and intergaps to the components
            mergegaps = numpy.asarray([max(intragaps[i], intergaps[i])\
                                       for i in range(len(intergaps))])
            pointers = numpy.where(mergegaps > threshold)[0]
            pept_split = [[]]
            for i, pept_ in enumerate(peptide.components):
                pept_split[-1].append(self.peptides[pept_.key])
                if i in pointers:
                    pept_split.append([])
            # rebuild the corresponding peptides:
            out_peptides = []
            for pept_ in pept_split:
                out_peptides.append(self.merge(*pept_))
            print("Splitting peptide: %s to %s" % (peptide.key, [p.key for p in out_peptides]))
            return out_peptides
        else:
            return [peptide]

    def get_peptide_alignment(self):
        """
        Get the alignment alignment object for each peptide individually.
        """
        peptides = {}  # Dictionnary containing all peptides in both order
        for i, pept in enumerate(self.peptides):
            # Try the two order for the coordinates
            for order in [1, -1]:
                coords = pept.coords[::order]
                aln = map_align(coords, self.gmap, n_iter=1, gap_o=-1., gap_e=-1)
                peptides[(i, order)] = Peptide(coords, aln, (i, order))
                self._get_coverage(peptides[(i, order)])
        self.peptides = peptides

    def align_fragments(self, max_iter=numpy.inf):
        """
        Compute the best alignment of the fragments to self.gmap
        • max_iter: Maximum number of iterations
        """
        # Compute the alignment scores for pairwise merging
        n_pept = len(self.peptides)
        for i, pept1 in enumerate(self.peptides.values()):
            print("Pairwise merging: %d/%d" % (i + 1, n_pept))
            self.get_neighbors(pept1)
            for pept2 in pept1.neighbors_r:
                self.merge(pept1, pept2)
            for pept2 in pept1.neighbors_l:
                self.merge(pept2, pept1)
        # Subsequent merging operations
        exclusion_set = []
        max_score = -numpy.inf
        #for _ in range(n_iter):
        n_iter = 0
        while len(exclusion_set) < len(self.adjmat.scores):
            pept1 = self.adjmat.max_score(exclusion_set=exclusion_set)
            exclusion_set.append(pept1.key)
            if pept1.score > max_score:
                max_score = pept1.score
            self.get_neighbors(pept1)
            for pept2 in pept1.neighbors_r:
                self.merge(pept1, pept2)
            for pept2 in pept1.neighbors_l:
                self.merge(pept2, pept1)
            print("%d/ %d: %s: score/score_max: %.2f/%.2f coverage: %.2f"\
                      %(len(exclusion_set), len(self.adjmat.scores),
                        str(pept1.key), pept1.score, max_score, pept1.coverage))
            n_iter += 1
            if n_iter >= max_iter:
                print("Maximum number of iterations (%d) reached" % max_iter)
                break
        self.coords_best = self.adjmat.max_score().coords
        self.aln = map_align(self.coords_best, self.gmap, fasta_file=self.fasta_file)

    def optimize_ca(self, outfilename=None):
        """
        Add missing CA on the CA trace after alignment.
        Optimize the position of the CA using the EM density.
        • outfilename: Filename for the output fasta file name containing the
                       alignment.
        """
        i = 0  # pointer for sequence 2
        pos1 = 0
        insertions = []  # list of beads to insert
        deletions = []  # list of beads to remove
        npoints = []  # The number of points to insert each time
        pointer1 = 0  # pointer for seq 1
        pointer2 = 0  # pointer for seq 2
        for pos1 in self.aln.alignment:
            pos2 = self.aln.alignment[pos1]
            while pos1 > pointer1:
                print("Deleting bead %d" % pointer1)
                deletions.append(pointer1)
                pointer1 += 1
            n_ca = 0  # Number of CA to add
            start = pos1 - 1  # where to add ca
            while pos2 > pointer2:
                pointer2 += 1
                n_ca += 1
            end = pos1  # where to add ca
            if n_ca > 0:
                print("Adding %d beads between %d and %d" % (n_ca, start, end))
                if start == -1:
                    # Instead of inserting before the first CA insert between
                    # the first one and the second one
                    start += 1
                    end += 1
                insertions.append(start)
                npoints.append(n_ca)
            pointer1 += 1
            pointer2 += 1
        self.coords_best = self.reshufle(insertions, npoints, deletions)
        # Refine the position of the added CA:
        graph = self.get_topology()
        fit = em_fit.Fit(self.emd,
                         self.level,
                         graph,
                         self.coords_best,
                         20000,
                         alpha_0=.1,
                         alpha_1=0.0,
                         radius_0=1.,
                         radius_1=0.,
                         refine=True)
        fit.learn()
        self.coords_best = fit.coords
        self.aln = map_align(self.coords_best, self.gmap, fasta_file=self.fasta_file, outfilename=outfilename)

    def get_topology(self):
        """
        Build the topology (adjacency matrix) for the current CA trace
        (self.coords)
        Return a graph object
        """
        n_ca = self.coords.shape[0]
        adjmat = numpy.ones((n_ca, n_ca)) * numpy.inf
        for i, j in zip(range(n_ca), range(1, n_ca)):
            adjmat[i, j] = 1.
            adjmat[j, i] = 1.
        graph = Graph.Graph(adjmat)
        graph.minimum_spanning_tree = adjmat
        return graph

    def reshufle(self, insertions, npoints, deletions):
        """
        insert linear interpolated points in self.coords
        • insertions: position of the insertions. The data are inserted before
        the positions given
        • npoints: the number of points to insert each time
        • deletions: list of beads to delete
        """
        offset = 0
        coords = numpy.copy(self.coords)
        total = coords.shape[0]
        i = -1
        seq_coverage_prev = float(total) / len(self.aln.sequence)
        n_beads = total + sum(npoints) - len(deletions)
        seq_coverage = float(n_beads) / len(self.aln.sequence)
        if seq_coverage > 1.:
            # More beads than residues:
            # Remove some beads at the end
            # Number of beads to delete at the end
            n_end_delete = n_beads - len(self.aln.sequence)
            bead_id = total - 1
            for _ in range(n_end_delete):
                while bead_id in deletions:
                    bead_id -= 1
                deletions.append(bead_id)
                print("Deleting ending bead %d (--%d)" % (bead_id, total - 1))
        for pos in range(total):
            if pos in insertions:
                i += 1
                n_ca = npoints[i]
                start = pos
                end = pos + 1
                a = self.coords[start]
                b = self.coords[end]
                data = EMDensity.interpolate(a, b, n_ca + 2)[1:-1]
                coords = numpy.insert(coords, end + offset, data, axis=0)
                offset += n_ca
            if pos in deletions:
                coords = numpy.delete(coords, pos + offset, axis=0)
                offset -= 1
        n_beads = coords.shape[0]
        seq_coverage = float(n_beads) / len(self.aln.sequence)
        print("Sequence coverage: %.2f -> %.2f" % (seq_coverage_prev, seq_coverage))
        return coords
