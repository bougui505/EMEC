#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2018-03-15 13:04:50 (UTC+0100)

import bisect
import sys
import numpy
import map_align
import scipy.spatial.distance

class Pool(object):
    """
    Store all the coordinates of the fragments
    """
    def __init__(self, fragments, cmap, fasta_sequence, n_iter=1., gap_o=-3.,
                 gap_e=-0.1, threshold=8., max_iter=numpy.inf):
        """
        • fragments: list of coordinates of the fragments
        • cmap: File containing the predicted contacts, or numpy.ndarray
        • fasta_sequence: sequence associated with the contact map in FASTA format
        • n_iter: number of iterations for the map_align algorithm
        • gap_o: gap opening penalty for the map_align algorithm
        • gap_e: gap extension penalty for the map_align algorithm
        • threshold: distance threshold to define a contact
        • max_iter: Maximum number of iterations for peptide alignments merging
        """
        self.fragments = fragments
        self.threshold = threshold
        #self._fids = [] # List of fragments ids (id, order)
        #for i in range(len(fragments)):
        #    self.fids.append( [(i, 1), ] )
        #    self.fids.append( [(i, -1), ] )
        if type(cmap) is numpy.ndarray:
            self.cmap = cmap
        else:
            self.cmap = map_align.read_gremlin(cmap)
        #n = self.cmap.shape[0]
        #self.cmap += numpy.eye(n)
        #self.cmap += numpy.eye(n, k=1)
        self.cmap[self.cmap>0] = 1
        self.sequence = map_align.read_sequence(fasta_sequence)
        self.gap_o = gap_o
        self.gap_e = gap_e
        self.n_iter = n_iter
        self.min_sep = numpy.abs(numpy.diff(numpy.asarray(numpy.where(self.cmap>0)), axis=0)).min() # Minimum contact residue range
        self.adjmat = self._get_pairwise_contacts()
        #self._scores = [] # Store all the alignment scores
        self._alignments = [] # Store all the alignments

    def _get_pairwise_contacts(self):
        """
        Compute the pairwise contact matrix between fragments:
        • For all (i,j); i!=j: total number of inter contacts between fragments
        i and j
        • For diagonal elements (i, i): total number of intra contact for
        fragment i
        """
        adjacency_matrix = numpy.zeros((len(self.fragments),)*2, dtype=int)
        diag = []
        for i, fragment1 in enumerate(self.fragments):
            pdist = scipy.spatial.distance.pdist(fragment1)
            pdist = scipy.spatial.distance.squareform(pdist)
            # Delete short range contacts
            pdist = numpy.triu(pdist, k=self.min_sep-1)
            pdist[pdist==0] = numpy.inf
            n_intra = (pdist <= self.threshold).sum()
            diag.append(n_intra)
            for j, fragment2 in enumerate(self.fragments[i:]):
                cdist = scipy.spatial.distance.cdist(fragment1, fragment2)
                n_inter = (cdist <= self.threshold).sum()
                adjacency_matrix[i, i+j] = n_inter
                adjacency_matrix[i+j, i] = n_inter
        diag = numpy.asarray(diag)
        #adjacency_matrix += diag
        #adjacency_matrix += diag[:, None]
        numpy.fill_diagonal(adjacency_matrix, diag)
        return adjacency_matrix

    @property
    def n_fragments(self):
        """
        Total number of fragments
        """
        return len(self._alignments)

    def get_n_contacts(self, fid):
        """
        Get the total number of contacts for the given fragment with fid
        """
        fids = [_[0] for _ in fid]
        n_intra, n_inter = 0, 0
        for i, fid1 in enumerate(fids):
            n_intra += self.adjmat[fid1, fid1]
            for fid2 in fids[i+1:]:
                n_inter += self.adjmat[fid1, fid2]
        return n_intra + n_inter

    def _merge_index(self, fid1, fid2):
        """
        Merge the two fragment index fid1 and fid2.
        E.g.
        • fid1: ((12, 1), )
        • fid2: ((16, -1), (15 ,1))
        Returns: ((12, 1), (16, -1), (15, 1))
        """
        merged = list(fid1)
        merged.extend(list(fid2))
        merged = tuple(merged)
        return merged

    def get_neighbor_score(self, fid1, fid2):
        """
        Get the neighboring score for the given two fragments
        Score = n_contacts_1_2 + aln_score_1 + aln_score_2
        """
        merged = self._merge_index(fid1, fid2)
        score = self.get_n_contacts(merged)
        score = float(score)
        db = self._db
        if fid1 in db:
            score += db[fid1].score
        if fid2 in db:
            score += db[fid2].score
        return score

    def get_neighbors(self, fid):
        """
        Get the neighbors of fid
        fid: [(i, 1), (j, -1), (k, 1), ...]
        """
        neighbors = set()
        # Get the simple neighboring fragemts
        fids_inp = [_[0] for _ in fid]
        for fid in fids_inp:
            neighbors |= set(numpy.where(self.adjmat[fid,:] > 0)[0])
        neighbors -= set(fids_inp)
        neighbor_list = []
        for n in neighbors:
            neighbor_list.extend([((n, 1), ), ((n, -1), )])
        # Get the already combined neighboring fragments
        for fid in self.fids[::-1]:
            fids = set([_[0] for _ in fid])
            if len(neighbors & fids) > 0 and len(fids & set(fids_inp)) == 0:
                neighbor_list.append(fid)
        return neighbor_list

    def _pairwise_merging(self):
        """
        Compute the alignmnents of the fragments pairwisely
        """
        n_fragments = len(self.fragments)
        neighbors_list = []
        for fid1_ in range(n_fragments):
            neighbors_list.append(self.get_neighbors( ((fid1_, 1),) ))
        for fid1_ in range(n_fragments):
            neighbors = neighbors_list[fid1_]
            for order in (1, -1):
                fid1 = ((fid1_, order),)
                for fid2 in neighbors:
                    fid = self._merge_index(fid1, fid2)
                    aln = self.get(fid)
                    print 'map_align: %s; score=%.2f'%(fid, aln.score)
                    fid = self._merge_index(fid2, fid1)
                    aln = self.get(fid)
                    print 'map_align: %s; score=%.2f'%(fid, aln.score)

    def merge(self):
        """
        Merge the fragment with the best score with the best neighbor
        """
        fid = self.fids[-1]

    def get_coords(self, fids):
        """
        Get the coordinates for the given fragment id (fid) and order
        fids: [(i, 1), (j, -1), (k, 1), ...]
        """
        coords = []
        for fid, order in fids:
            coords.extend(list(self.fragments[fid][::order]))
        coords = numpy.asarray(coords)
        return coords

    @property
    def scores(self):
        """
        Return all the alignment score values sorted
        """
        return [_.score for _ in self._alignments]

    @property
    def fids(self):
        """
        Return all the fragment ids sorted by score values
        """
        return [_.fid for _ in self._alignments]

    @property
    def _db(self):
        """
        Return the mapping {fid:aln, ...}
        """
        return dict(zip(self.fids, self._alignments))

    def _store(self, aln):
        """
        Store the given alignment to self._scores, self._alignments and
        self._fids.
        """
        ind = bisect.bisect_left(self.scores, aln.score)
        self._alignments.insert(ind, aln)

    def get(self, fids):
        """
        Get the contact map alignment for the given fragment id (fid) and order
        fids: ((i, 1), (j, -1), (k, 1), ...)
        """
        db = self._db
        if fids in db:
            return db[fids]
        else:
            coords = self.get_coords(fids)
            aln = map_align.map_align(coords, self.cmap, gap_o=self.gap_o,
                                      gap_e=self.gap_e, n_iter=self.n_iter,
                                      threshold=self.threshold)
            aln.sequence = self.sequence
            aln.get_seq()
            aln.fid = fids
            self._store(aln)
            return aln
