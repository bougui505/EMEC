#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-12-07 16:41:21 (UTC+0100)

import numpy
import scipy.spatial.distance
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/bougui/lib/MAST')
import map_align
import all_atoms

"""
Simple script to compare two contact maps
"""
def get_cmap_from_pdb(pdbfile, threshold, CA=True):
    atom_types, _, _, resids, coords, _ = all_atoms.read_pdb(pdbfile)
    resids = numpy.asarray(resids)
    if CA:
        selection_CA = numpy.asarray(atom_types) == 'CA'
        coords = coords[selection_CA]
        resids = resids[selection_CA]
    dmap = scipy.spatial.distance.pdist(coords)
    dmap = scipy.spatial.distance.squareform(dmap)
    resid_list = numpy.unique(resids)
    cmap = numpy.ones((resid_list[-1],)*2)*numpy.nan
    for i, resid_i in enumerate(resid_list):
        sys.stderr.write('%d/%d\r'%(i+1, len(resid_list)))
        sys.stderr.flush()
        for j, resid_j in enumerate(resid_list[i:]):
            j = j + i
            selection_i = (resids == resid_i)
            selection_j = (resids == resid_j)
            mindist = (dmap[selection_i][:, selection_j]).min()
            cmap[resid_i-1, resid_j-1] = mindist
            cmap[resid_j-1, resid_i-1] = mindist
    cmap = cmap <= threshold
    return cmap

def get_cmap(infile, threshold, CA=True):
    try:
        cmap = map_align.read_gremlin(infile)
    except (IndexError, ValueError):
        cmap = get_cmap_from_pdb(infile, threshold, CA=CA)
    return cmap

def scatter_cmap(cmap, marker='.', label=None, aln=None):
    x, y = numpy.where(cmap == 1)
    if aln is not None:
        new_x, new_y = [],[]
        for x_, y_ in zip(x, y):
            x_ = aln.alignment.get(x_)
            y_ = aln.alignment.get(y_)
            if x_ is not None and y_ is not None:
                new_x.append(x_)
                new_y.append(y_)
        x = new_x
        y = new_y
    plt.scatter(x, y, marker=marker, label=label)

def align_cmaps(cmap, cmap_ref, threshold):
    aln = map_align.map_align(cmap, cmap_ref, threshold=threshold, gap_o=-3, gap_e=0, n_iter=20)
    if aln is not None:
        score = aln.score
    else:
        score = -numpy.inf
    #print "Alignment score: %.2f"%score
    return aln

if __name__ == '__main__':
    infile = sys.argv[1]
    infile_ref = sys.argv[2]
    # if negative threshold contact map on CA
    threshold = float(sys.argv[3])
    align_map = bool(int(sys.argv[4]))
    if threshold < 0:
        CA=True
        threshold = -threshold
    else:
        CA=False
    cmap = get_cmap(infile, threshold, CA=CA)
    cmap_ref = get_cmap(infile_ref, threshold, CA=CA)
    if align_map:
        aln = align_cmaps(cmap, cmap_ref, threshold=threshold)
    else:
        aln = None
    scatter_cmap(cmap_ref, marker='o', label=infile_ref)
    scatter_cmap(cmap, label=infile, aln=aln)
    if aln is not None:
        plt.title('Alignment score: %.2f'%(aln.score+aln.npos-aln.n_align))
    plt.legend()
    plt.show()
