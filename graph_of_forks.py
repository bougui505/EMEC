#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-06-02 09:22:02 (UTC+0200)

import Graph
import Tree
import skeleton
import EMDensity
import numpy
import scipy.spatial
import networkx as nx

def get_nodes(graph):
    """
    Get the list of nodes of the given graph
    """
    nodes = []
    for k1 in graph:
        if k1 not in nodes:
            nodes.append(k1)
        for k2 in graph[k1]:
            if k2 not in nodes:
                nodes.append(k2)
    nodes.sort()
    return numpy.asarray(nodes)

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
    atom_types = numpy.asarray([e.strip() for e in splitted_lines[:, 2]])
    aa_list = [e.strip() for e in splitted_lines[:, 3]]
    chain_ids = numpy.asarray([e.strip() for e in splitted_lines[:, 4]])
    resids = numpy.asarray([int(e) for e in splitted_lines[:, 5]])
    # Check if the given pdbfile is a CA trace only
    is_ca_trace = len(set(resids)) == len(resids)
    coords = numpy.float_(splitted_lines[:, 6:9])
    coords_per_chain = [coords[chain_ids == chainid] for chainid in numpy.unique(chain_ids)]
    ca_traces = coords[atom_types == 'CA']
    chain_ids = chain_ids[atom_types == 'CA']
    resids = resids[atom_types == 'CA']
    #resids = [resids[chain_ids == chainid] for chainid in numpy.unique(chain_ids)]
    ca_trace = [ca_traces[chain_ids == chainid] for chainid in numpy.unique(chain_ids)]
    mapping = dict(zip(numpy.unique(chain_ids),
                   numpy.arange(len(set(chain_ids)))))
    chain_ids = numpy.asarray([mapping[e] for e in chain_ids])
    resids = numpy.asarray(resids)
    return ca_traces, chain_ids, resids

class Forks(object):
    """
    Detect the forks in a graph and build a new weighted graph F:
    - The nodes of F are the forks of the given graph.
    - The edges are the connected forks in the given graph.
    - The weights are the number of nodes to join the forks.
    """
    def __init__(self, mstree, coords=None, ref=None, density=None):
        """
        • mstree: the adjacency matrix of the minimum spanning tree
        • coords (optional): coordinates associated to the nodes of the
                             adjacency matrix
        • ref (optional): reference pdb file name to compare with
        • density (optional): EM density object class
        """
        graph = Graph.Graph(mstree)
        graph.minimum_spanning_tree = mstree
        self.density = density
        self.tree = Tree.Tree(graph, coords=coords, density=density)
        self.forks, _ = skeleton.detect_forks(self.tree)
        self.F = None # The Forks graph
        # chain attribution of the nodes relative to ref
        self.chainids = None
        self.resids = None
        if ref is not None and coords is not None:
            ref, chainids, resids = read_pdb(ref)
            kdtree = scipy.spatial.KDTree(ref)
            _, nn = kdtree.query(coords)
            self.chainids = chainids[nn]
            self.resids = resids[nn]
        self.F = self.tree.get_forks_graph()

    def plot_graph(self):
        """
        Plot the graph of forks using networkx
        """
        nxG_mod = nx.Graph()
        for n1 in self.F:
            for n2 in self.F[n1]:
                nxG_mod.add_edge(n1, n2,weight=self.F[n1][n2])
        if self.density is not None:
            edge_labels = {(u, v): "%.1f"%d['weight'] for u, v, d in nxG_mod.edges(data=True)}
        else:
            edge_labels = {(u, v): d['weight'] for u, v, d in nxG_mod.edges(data=True)}
        # Draw the Graph for the reference protein is given:
        if self.resids is not None:
            nxG_ref = nx.DiGraph()
            for chainid in numpy.unique(self.chainids):
                nodes = numpy.asarray([_ for _ in get_nodes(self.F)\
                                       if self.chainids[_] == chainid])
                resids = [self.resids[_] for _ in nodes]
                sorter = numpy.argsort(resids) # Peptidic chain
                sorted_nodes = nodes[sorter]
                for (n1, n2) in zip(sorted_nodes, sorted_nodes[1:]):
                    nxG_ref.add_edge(n1, n2)
            nxG = nx.compose(nxG_mod, nxG_ref)
        else:
            nxG = nxG_mod
        pos = nx.graphviz_layout(nxG)
        # Drawing:
        if self.chainids is not None:
            colors = [self.chainids[k] for k in nxG_mod]
            nx.draw(nxG_mod, pos=pos, node_color=colors, node_size=80)
        else:
            nx.draw(nxG_mod, pos=pos, node_size=80)
        nx.draw_networkx_edge_labels(nxG_mod, pos, edge_labels=edge_labels)
        if self.resids is not None:
            nx.draw(nxG_ref, pos=pos, node_size=1., arrows=False, width=3.)
