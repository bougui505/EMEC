#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-01-26 11:19:08 (UTC+0100)

import EMDensity
import scipy.spatial
import Graph
import numpy
import Tree
import scipy.sparse
import scipy.signal
import progress_reporting
import os
import optimizer
#import graph_of_forks
#import matplotlib.pyplot as plt


def flatten_nested_list(listoflist):
    """
    Flatten a nested list of list:
    [[1, 2, 3], [7, 8, 9]] -> [1, 2, 3, 7, 8, 9]
    """
    out = []
    for list_ in listoflist:
        out.extend(list_)
    return out


def print_pymol_selection(nodes):
    """
    Print pymol friendly selection
    """
    print('select resid ' + '+'.join([str(e + 1) for e in nodes]))


def add_to_coo(coo, data, row, col):
    """
    Add the given data with the given COOrdinates to the given coo matrix (coo).
    Return a copy of coo with added data
    """
    row = numpy.r_[coo.row, row]
    col = numpy.r_[coo.col, col]
    data = numpy.r_[coo.data, data]
    return scipy.sparse.coo_matrix((data, (row, col)), shape=coo.shape)


def get_connected_components(mstree, coords, density, pruning_threshold, save_npz=False):
    """
    Get each connected components (the forest)
    • save_npz: If True save the MS-trees in a npz file
    """
    graph = Graph.Graph(adjacency_matrix=mstree)
    graph.minimum_spanning_tree = mstree
    tree = Tree.Tree(graph)
    _, forest = tree.find_subtrees(return_forest=True)
    chains = []
    for tree_id, comp in enumerate(forest):
        selection = [e.index for e in comp]
        mstree_, coords_ = Tree.select_tree(graph.minimum_spanning_tree, selection, coords=coords)
        pdist = scipy.spatial.distance.pdist(coords_)
        pdist = scipy.spatial.distance.squareform(pdist)
        graph_ = Graph.Graph(mstree_)
        graph_.minimum_spanning_tree = mstree_
        tree_ = Tree.Tree(graph_)
        diameter = tree_.get_diameter(pdist)
        if diameter >= 5 * 3.8:  # Select Tree with at least pentapeptides
            if save_npz:
                npz = "mstree_%d.npz" % tree_id
            else:
                npz = None
            chains.append(Chain(mstree_, coords_, density, pruning_threshold, npz=npz))
    return chains


def detect_forks(tree):
    """
    detect remaining nodes of degree > 2 in the given tree
    """
    mstree = tree.graph.minimum_spanning_tree
    forks = numpy.where(Tree.get_degree(mstree) > 2)[0]
    degrees = Tree.get_degree(mstree)[forks]
    return forks, degrees


class Chain(object):
    """
    Class containing a polypeptidic chain object
    """
    def __init__(self, mstree, coords, density, pruning_threshold=10, npz=None):
        """
        mstree: minimum spanning tree giving the topology of the chain
        coords: coordinates associated with the topology
        density: EM density object class
        pruning_threshold: threshold for pruning fragment. If the length of a
        fragment to prune is less than pruning_threshold, remove it, else keep
        it and just remove the forked node.
        """
        self.topology = mstree
        self.coords = coords
        graph = Graph.Graph(mstree)
        graph.minimum_spanning_tree = mstree
        self.tree = Tree.Tree(graph)
        self.vinepaths = self.tree.get_vine_paths()
        self.backbone, self.sidechains = self.get_backbone_sidechains()
        if npz is not None:
            numpy.savez(npz, mstree=self.backbone, coords=self.coords)
            pdbfilename = '%s.pdb' % os.path.splitext(npz)[0]
            optimizer.write_minimum_spanning_tree(self.coords, self.backbone, outfile=pdbfilename)
        # Clean up the backbone
        self.density = density
        self.pruning_threshold = pruning_threshold
        self.clean_backbone()
        # protein is the backbone and sidechains
        self.protein = self.get_protein()

    def get_backbone_sidechains(self):
        """
        Get the most probable backbone
        """
        backbone_selection = set(range(self.tree.n_nodes))
        sidechains_selection = set()
        for vpath in self.vinepaths.values():
            # Attribute the shortest vine path to the sidechain
            l = numpy.asarray([len(e) for e in vpath])
            sidechains_selection |= set(vpath[l.argsort()[0]])
        backbone_selection -= sidechains_selection
        backbone = self.tree.substract_from_tree(sidechains_selection, mutate=False)
        # Add the CA to sidechains for connectivity
        backbone_selection -= set(self.vinepaths.keys())
        sidechains = self.tree.substract_from_tree(backbone_selection, mutate=False)
        print(sidechains)
        # Remove edges present both in the backbone and the sidechains:
        graph_sidechains = Graph.Graph(sidechains)
        graph_sidechains.minimum_spanning_tree = sidechains
        tree_sidechains = Tree.Tree(graph_sidechains)
        sidechains = tree_sidechains.substract_from_tree(numpy.where(sidechains == backbone), mutate=False)
        return backbone, sidechains

    def clean_backbone(self):
        """
        Automatically remove forks in the backbone.
        Select the shortest vine paths
        """
        graph = Graph.Graph(self.backbone)
        graph.minimum_spanning_tree = self.backbone
        tree = Tree.Tree(graph, density=self.density, coords=self.coords)
        forks, degrees_fork = detect_forks(tree)
        print("%d forks: %s" % (len(forks), forks))
        ######################
        #        i = 0
        #        F = graph_of_forks.Forks(self.backbone, self.coords, '4ci0.pdb', density=self.density)
        #        plt.figure(figsize=(12, 12))
        #        F.plot_graph()
        #        plt.savefig('forks_graph_%d.png'%i)
        #        plt.clf()
        ######################
        while len(forks) > 0:
            tree.get_forks_graph()
            fork = tree.forks_score.keys()[0]  # Fork with the highest score
            # Get the path to remove
            torm = sorted(tree.P[fork].items(), key=lambda t: len(t[1]))[0][1]
            if len(torm) < self.pruning_threshold:  # Short fragment, remove it
                print("Resolving fork %d: 1/%d" % (
                    fork,
                    len(forks),
                ))
                tree.substract_from_tree(torm)
            else:  # Bigger one, keep it and remove the fork node
                print("Deleting fork %d: 1/%d" % (
                    fork,
                    len(forks),
                ))
                tree.substract_from_tree([
                    fork,
                ])
            self.backbone = tree.graph.minimum_spanning_tree
            forks, degrees_fork = detect_forks(tree)


######################
#            i += 1
#            F = graph_of_forks.Forks(self.backbone, self.coords, '4ci0.pdb', density = self.density)
#            plt.figure(figsize=(12, 12))
#            F.plot_graph()
#            plt.savefig('forks_graph_%d.png'%i)
#            plt.clf()
######################

    def get_protein(self):
        """
        The protein is the union of self.backbone and self.sidechain
        """
        mstree = numpy.copy(self.backbone).item()
        graph = Graph.Graph(mstree)
        graph.minimum_spanning_tree = mstree
        tree = Tree.Tree(graph)
        tree.add_to_tree(self.sidechains)
        protein = tree.graph.minimum_spanning_tree
        return protein

    def get_ca(self, sigma):
        """
        Get the most probable CA positions based on the distance between
        consecutive CA and the degrees of the nodes (the CA nodes should be of
        degree 3)
        • sigma: standard deviation on the CA-CA distance
        Amino acid geometry:
        See: https://goo.gl/LbgjLI
        CA-CA: 3.8 A
        CA-C: 1.53 A
        C-N 1.32 A
        N-CA: 1.47 A
        through bond CA-CA distance: 4.32 A
        """
        def gaussian(x, mu, sigma):
            return numpy.exp(-(x - mu)**2 / (2 * sigma**2))

        graph = Graph.Graph(self.protein)
        graph.minimum_spanning_tree = self.protein
        tree = Tree.Tree(graph)
        degrees = Tree.get_degree(self.protein)
        roots = numpy.where(degrees == 3)[0]
        pdist = scipy.spatial.distance.pdist(self.coords)
        pdist = scipy.spatial.distance.squareform(pdist)
        n = len(roots)
        topdist = numpy.zeros((n, ) * 2)  # matrix containing topological distances
        for i, root_i in enumerate(roots):
            bfs = tree.graph.bfs(root_i, pdist=pdist)
            bfs = dict([(e.index, e.distance) for e in bfs])
            for j, root_j in enumerate(roots):
                topdist[i, j] = bfs[root_j]
                topdist[j, i] = bfs[root_j]
        pmat = gaussian(topdist, 4.32, sigma)
        pointers = numpy.where(pmat.max(axis=0) >= numpy.exp(-1. / 2))[0]
        # Compute the adjacency matrix for the CA
        row, col, data = [], [], []
        for i in pointers:
            j = numpy.arange(degrees.size)[degrees == 3][pmat[i].argmax()]
            resid = numpy.arange(degrees.size)[degrees == 3][i]
            row.append(resid)
            col.append(j)
            data.append(1.)
            row.append(j)
            col.append(resid)
            data.append(1.)
        self.CA = scipy.sparse.coo_matrix((data, (row, col)), shape=self.protein.shape)
        self.CA = Tree.reorder_vine_paths(self.CA, self.backbone)
        # Get the ids of the CA
        resids = numpy.arange(degrees.size)[degrees == 3][pointers]
        # Add sparse CA (CA not connected to any other CA):
        topdist[topdist == 0] = numpy.inf
        pointers = numpy.where((topdist > 4.32).all(axis=0))[0]
        resids = set(resids)
        sparse_resids = numpy.arange(degrees.size)[degrees == 3][pointers]
        # Remove already found resids from sparse_resids
        sparse_resids = set(sparse_resids)
        sparse_resids -= resids
        sparse_resids = numpy.asarray(list(sparse_resids))
        # Add sparse resids to the CA adjacency matrix (diagonal elements)
        self.CA = add_to_coo(self.CA, ([
            1.,
        ] * len(sparse_resids), (sparse_resids, sparse_resids)))
        resids |= set(sparse_resids)
        resids = numpy.asarray(list(resids))
        # Remove the sidechains for the non detected CA:
        graph = Graph.Graph(self.sidechains)
        graph.minimum_spanning_tree = self.sidechains
        tree = Tree.Tree(graph)
        # List of all possible CA
        roots = set(numpy.where(Tree.get_degree(self.protein) == 3)[0])
        # Remove the list of CA to keep
        roots -= set(resids)
        torm = []
        for root in roots:
            vpath = tree.get_vine_path(root)
            torm.extend(vpath)
        tree.substract_from_tree(torm)
        # Compute the new protein with sidechains
        self.protein = self.get_protein()

    def get_missing_ca(self):
        """
        Return segment of the backbone with no CA attributed
        """
        graph = Graph.Graph(self.backbone)
        graph.minimum_spanning_tree = self.backbone
        tree_bb = Tree.Tree(graph)

        graph = Graph.Graph(self.CA)
        graph.minimum_spanning_tree = self.CA
        tree_ca = Tree.Tree(graph)

        bb = tree_bb.get_vine_paths().values()[0][0]
        bb_ca = Tree.intersect(tree_bb, tree_ca)
        no_ca = list(set(bb) - set(bb_ca))

        peptides = tree_bb.substract_from_tree(bb_ca, mutate=False)
        graph = Graph.Graph(peptides)
        graph.minimum_spanning_tree = peptides
        tree = Tree.Tree(graph)
        vpaths = tree.get_vine_paths()

        ca_list = tree_ca.get_vine_paths().keys()
        selected = set()
        paths = []
        for ca in ca_list:
            neighbors = tree_bb.G[ca].keys()
            for n in neighbors:
                path = []
                if n in vpaths and n not in selected:
                    path.extend([ca, n])
                    path.extend(vpaths[n][0][::-1])
                    selected |= set(path)
                if len(path) > 0:
                    paths.append(path)
        return paths

    def add_ca(self, sigma):
        """
        Add the missing CA based on distances. The missing CA are CA with no
        sidechain detected.
        • sigma: standard deviation for the CA position
        """
        paths = self.get_missing_ca()

        def gaussian(x, mu, sigma):
            return numpy.exp(-(x - mu)**2 / (2 * sigma**2))

        def gaussians(x, n):
            return numpy.squeeze(numpy.asarray([[gaussian(x, i * 4.32, sigma)] for i in range(1, n + 1)]).sum(axis=0))

        selection = []  # list of pointers for new CA
        for path in paths:
            path = numpy.asarray(path)
            distances = numpy.linalg.norm(numpy.diff(self.coords[path], axis=0), axis=1).cumsum()
            nca = int(round(distances[-1] / 4.32))
            if nca > 0:  # CA to add...
                p_ca = gaussians((distances, nca))
                pointers = path[scipy.signal.argrelmax(p_ca, mode='wrap')[0]]
                selection.extend(pointers)
        # The list of all CA:
        ca_list = list(numpy.where(Tree.get_degree(self.CA) > 0)[0])
        ca_list.extend(selection)
        # Create the new adjacency matrix for all the CA:
        graph = Graph.Graph(self.backbone)
        graph.minimum_spanning_tree = self.backbone
        tree_bb = Tree.Tree(graph)
        self.CA = tree_bb.union(ca_list)

    @property
    def coords_CA(self):
        """
        coordinates of the CA atoms
        """
        graph = Graph.Graph(self.CA)
        graph.minimum_spanning_tree = self.CA
        tree = Tree.Tree(graph)
        start, _ = tree.find_subtrees()[0]
        vpath = tree.get_vine_path(start)
        return self.coords[vpath]

    def sidechain(self, ca):
        """
        Return the sidechain for the given C-alpha atom id
        """
        graph = Graph.Graph(self.sidechains)
        graph.minimum_spanning_tree = self.sidechains
        tree = Tree.Tree(graph)
        try:
            sc = tree.get_vine_path(ca)[1:]
        except KeyError:
            sc = None
        return sc

    def CB(self):
        """
        Return the list of C-beta atom ids
        """
        ca_list = numpy.where(Tree.get_degree(self.protein) == 3)[0]

        def gaussian(x, mu, sigma):
            return numpy.exp(-(x - mu)**2 / (2 * sigma**2))

        cb_list = []
        for ca in ca_list:
            sc = self.sidechain(ca)
            if sc is not None:
                casc = [
                    ca,
                ]  # CA and SideChain
                casc.extend(sc)
                coords = self.coords[casc]
                distances = numpy.sqrt(((numpy.diff(coords, axis=0))**2).sum(axis=1))
                distances = numpy.cumsum(distances)
                probs = gaussian(distances, 1.5, .1)
                cb = sc[probs.argmax()]  # index of the C-beta
                cb_list.append(cb)
        return cb_list


class Skeleton(object):
    """
    Skeleton of an electron microscopy density map
    """
    def __init__(self, emd, level, pruning_threshold):
        """
        • emd: Electron microscopy (EM) density file in netCDF (nc) format
        • level: Thresholding level to apply to the EM map
        """
        self.emd = EMDensity.Density(emd, level)
        selection = self.emd.density > 0.
        self.density = self.emd.density[selection]
        xgrid = self.emd.xgrid[selection]
        ygrid = self.emd.ygrid[selection]
        zgrid = self.emd.zgrid[selection]
        self.grid = numpy.c_[xgrid, ygrid, zgrid]
        self.kdtree = scipy.spatial.cKDTree(self.grid)
        self.step = numpy.linalg.norm([self.emd.x_step, self.emd.y_step, self.emd.z_step])
        self.adjmat = self.get_adjmat()
        self.graph = Graph.Graph(adjacency_matrix=self.adjmat)
        self.graph.get_minimum_spanning_tree()
        self.mstree = numpy.copy(self.graph.minimum_spanning_tree).item()
        self.coords = self.grid
        # Clean the mstree:
        self.clean_mstree()
        # Get the connected components (the polypeptidic chains)
        print("Detecting chains")
        chains = get_connected_components(self.mstree, self.coords, self.emd, pruning_threshold, save_npz=True)
        # The chain class clean the mstree, therefore the resulting mstree could
        # contain multiple connected components.
        # The following lines split the resulting chains
        self.chains = []
        sigma = numpy.mean([self.emd.x_step, self.emd.y_step, self.emd.z_step])
        progress = progress_reporting.Progress(len(chains), delta=1, label='Detecting chains')
        i = 0
        for chain in chains:
            chains_ = get_connected_components(chain.protein, chain.coords, self.emd, pruning_threshold)
            for chain_ in chains_:
                #if (~numpy.isinf(chain_.backbone)).sum() > 0:
                if Tree.get_degree(chain_.backbone).sum() > 0:
                    # Get the CA positions:
                    chain_.get_ca(sigma=sigma)
                    chain_.add_ca(sigma=sigma)
                    if Tree.get_degree(chain_.CA).sum() > 0:
                        self.chains.append(chain_)
                        i += 1
            progress.count(report="%d chains" % i)

    def get_adjmat(self):
        """
        Compute the adjacency matrix from the EM density
        """
        index = self.kdtree.query_ball_point(self.grid, self.step)
        row = []
        col = []
        data = []
        for i, index_ in enumerate(index):
            index_ = set(index_) - set([i])
            for j in index_:
                row.append(i)
                col.append(j)
                data.append(1 / (self.density[i] * self.density[j]))
        adjmat = scipy.sparse.coo_matrix((data, (row, col)), shape=(self.density.size, ) * 2)
        return adjmat

    def clean_mstree(self):
        """
        Clean the minimum spanning tree by removing vine paths
        n_iter: number of iteration
        """
        degree = max(Tree.get_degree(self.mstree))
        degree_prev = 0
        while degree != degree_prev:
            graph = Graph.Graph(adjacency_matrix=self.mstree)
            graph.minimum_spanning_tree = self.mstree
            tree = Tree.Tree(graph)
            selection = set(range(tree.n_nodes))
            vpaths = tree.get_vine_paths()
            # Keep only one (the longest) vine path per node.
            for node in vpaths:
                vpaths_ = numpy.asarray(vpaths[node])
                l = numpy.asarray([len(e) for e in vpaths_])
                torm = set(flatten_nested_list(vpaths_[numpy.argsort(l)][:-1]))
                # Delete vine paths of length 1
                torm |= set(flatten_nested_list(vpaths_[l == 1]))
                selection -= torm
            selection = list(selection)
            self.mstree, self.coords = Tree.select_tree(self.mstree, selection, coords=self.coords)
            degree_prev = degree
            degree = max(Tree.get_degree(self.mstree))
            print("Pruning tree (previous degree, new degree): %d, %d" % (degree_prev, degree))
