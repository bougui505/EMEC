#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-01-17 15:17:50 (UTC+0100)

import numpy
import Graph
import scipy.sparse
import collections
import scipy.spatial.distance

def get_degree(adjmat):
    """
    Return the vector of degrees for the given adjacency matrix
    """
    degrees = []
    adjmat = adjmat.tocsr()
    for i in range(adjmat.shape[0]):
        degrees.append(adjmat[i].indices.size)
    return numpy.asarray(degrees)


def select_tree(adjmat, selection, coords=None):
    """
    Select the nodes given by selection and return the corresponding
    adjacency matrix extracted from the given one (adjmat)
    • adjmat: adjacency matrix where to select nodes from (ndarray: n*n)
    • selection: 1D vector of boolean of size n, or list of node ids to select
    • coords: optional. Coordinates associated with the nodes. If not None,
    the selected coordinates are also returned.
    """
    test = selection[0]
    if (type(test) is bool or type(test) is numpy.bool_):
        # Convert boolean selection to a list of nodes
        selection = list(numpy.where(selection)[0])
    adjmat = adjmat.tocsr()
    row, col, data = [], [], []
    # Mapping of the old indexing to the new indexing
    mapping = dict(zip(selection, range(len(selection))))
    for i in selection:
        for j in adjmat[i].indices:
            if j in selection:
                row.append(mapping[i])
                col.append(mapping[j])
                data.append(adjmat[i, j])
    adjmat = scipy.sparse.coo_matrix((data, (row, col)), shape=(len(selection), ) * 2)
    if coords is None:
        return adjmat
    else:
        return adjmat, coords[selection]


def clean_tree(MSTree, coords, n_iter=1, exclusion_set=set()):
    """
    Remove edges of degree 1
    exclusion_set: Nodes to exclude from the selected set to clean
    """
    for i in range(n_iter):
        degrees = get_degree(MSTree)
        selection = set(numpy.where((degrees > 1))[0])
        selection -= exclusion_set
        selection = list(selection)
        MSTree, coords = select_tree(MSTree, selection, coords)
    return MSTree, coords


def reorder_vine_paths(mstree1, mstree2):
    """
    reorder the mstree_1 which could contain multiple vine paths so that it
    keeps the same order as the unique vpath contained in mstree2
    """
    graph_1 = Graph.Graph(mstree1)
    graph_1.minimum_spanning_tree = mstree1
    tree_1 = Tree(graph_1)
    forest = tree_1.find_subtrees()

    graph_2 = Graph.Graph(mstree2)
    graph_2.minimum_spanning_tree = mstree2
    tree_2 = Tree(graph_2)
    start, _ = tree_2.find_subtrees()[0]
    vpath = numpy.asarray(tree_2.get_vine_path(start))

    row, col, data = [], [], []
    for start, _ in forest:
        vpath_1 = numpy.asarray(tree_1.get_vine_path(start))
        if set(vpath_1).issubset(set(vpath)):
            sorter = numpy.argsort([numpy.where(vpath == e)[0][0] for e in vpath_1])
            vpath_1 = vpath_1[sorter]
            for i, j in zip(vpath_1, vpath_1[1:]):
                row.append(i)
                col.append(j)
                data.append(1.)
                row.append(j)
                col.append(i)
                data.append(1.)
    adjmat = scipy.sparse.coo_matrix((data, (row, col)),\
                                 shape=tree_1.graph.minimum_spanning_tree.shape)
    return adjmat


def intersect(tree1, tree2):
    """
    Get the intersection between two vine trees:
    E.G:
    tree1: A-B-C-D-E-F-G-H
    tree2: --B---D-----G-H
    intersect(tree1, tree2) -> B-C-D-G-H
    """
    terminus = tree1.find_subtrees()[0]
    vpaths_2 = tree2.get_vine_paths()
    intersection = []
    for end in vpaths_2:
        if len(vpaths_2[end][0]) > 0:
            start = vpaths_2[end][0][0]
            intersection.extend(tree1.get_vine_path(terminus[0], start=start, end=end))
        else:
            intersection.append(end)
    return intersection

def line_sphere_intersection(v, o, c, r, upper_dist):
    """
    Return the coordinates of the intersection between a line and a sphere:
    see: https://goo.gl/euF3eQ
    • v: unit vector of the direction of the line
    • o: origin of the line
    • c: center of the sphere
    • r: radius of the sphere
    • upper_dist: return only the point below the upper distance from o
    """
    A = -v.dot(o-c)
    B = (v.dot(o-c))**2 - numpy.linalg.norm(o-c)**2 + r**2
    if B >= 0:
        d1 = A + numpy.sqrt(B)
        d2 = A - numpy.sqrt(B)
        results = numpy.asarray([d1, d2])
        selection = numpy.logical_and(results > 0, results <= upper_dist)
        if selection.any():
            d = results[selection][0]
            return o + v*d
        else:
            return None
    else:
        return None

def resample_path(path_coords, delta):
    """
    Resample the path to get evenly distributed beads along the path with a
    given delta distance.
    • path_coords: coordinates of the beads along the path
    • delta: delta distance
    """
    # Distances along the path from the first bead
    #distances = numpy.linalg.norm(path_coords - path_coords[0], axis=1)
    distances = numpy.linalg.norm(numpy.diff(path_coords, axis=0), axis=1).cumsum()
    distances = numpy.r_[0, distances]
    n_beads = int(distances[-1]/delta) + 1
    distances_target = numpy.asarray([i*delta for i in range(n_beads)])
    delta_target = distances[:, None] - numpy.broadcast_to(distances_target, (distances.size, distances_target.size))
    delta_target[delta_target>0] = numpy.inf
    delta_target = numpy.abs(delta_target)
    bead_coords = [path_coords[0], ]
    bead_id = 0
    bead_coords_new = None
    for i in range(1, n_beads):
        while bead_coords_new is None and bead_id + 1 < path_coords.shape[0]:
            coords_low = path_coords[bead_id]
            coords_up = path_coords[bead_id+1]
            bead_coords_last = bead_coords[-1]
            v = coords_up - coords_low
            upper_dist = numpy.linalg.norm(v)
            v /= upper_dist
            bead_coords_new = line_sphere_intersection(v, coords_low, bead_coords_last, delta, upper_dist)
            bead_id += 1
        if bead_coords_new is not None:
            bead_coords.append(bead_coords_new)
        bead_coords_new = None
    return numpy.asarray(bead_coords)

def clean_path(fragments, delta):
    """
    Remove beads with a distance lower than delta from the other.
    • fragments: list of fragment coordinates
    • delta: threshold distance
    """
    coords = []
    for fragment in fragments:
        coords.extend(list(fragment))
    coords = numpy.asarray(coords)
    dmat = scipy.spatial.distance.pdist(coords)
    dmat = scipy.spatial.distance.squareform(dmat)
    numpy.fill_diagonal(dmat, numpy.inf)
    i = 0
    fragments_cleaned = []
    for fragment in fragments:
        fragment_cleaned = []
        for bead in fragment:
            if (dmat[i] >= delta).all():
                fragment_cleaned.append(bead)
            else:
                dmat[:, i] = numpy.inf
            i += 1
        if len(fragment_cleaned) > 1:
            fragments_cleaned.append(numpy.asarray(fragment_cleaned))
    return fragments_cleaned



class Tree(object):
    """
    A class to get trees from a graph
    """
    def __init__(self, graph, coords=None, density=None):
        """
        • graph: A graph object
        • coords (optional): 3D coordinates associated withe the nodes
        • density (optional): EM Density object class
        If density and coords are given the density is used to score the forks
        """
        self.graph = graph
        if self.graph.minimum_spanning_tree is None:
            self.graph.get_minimum_spanning_tree()
        self.n_nodes = self.graph.minimum_spanning_tree.shape[0]
        self.G = self.graph.get_graph(self.graph.minimum_spanning_tree)
        self.degrees = get_degree(self.graph.minimum_spanning_tree)
        self.forks = list(numpy.where(self.degrees > 2)[0])
        self.F = None  # The forks graph

        self.coords = coords
        self.density = density

    def get_bfs_distances(self, root):
        """
        Get the bfs distances from the root
        Return: {dist:[node1, node2, ...]}
        """
        bfs = self.graph.bfs(root)
        bfs_dist = {}
        for node in bfs:
            pointer, distance = node.index, node.distance
            if distance not in bfs_dist:
                bfs_dist[distance] = [
                    pointer,
                ]
            else:
                bfs_dist[distance].append(pointer)
        return bfs_dist

    def find_subtree(self, root, pdist=None):
        """
        Find the subtree rooted in root
        • pdist: pairwise distance matrix between nodes (shape: n*n, with n the
        number of nodes). If not None, use the distances given in the matrix to
        compute the distance from the root.
        """
        bfs1 = self.graph.bfs(root, pdist=pdist)
        start = bfs1[-1].index
        bfs2 = self.graph.bfs(start, pdist=pdist)
        return bfs2

    def get_diameter(self, pdist=None):
        """
        Return the diameter of the tree
        • pdist: pairwise distance matrix between nodes (shape: n*n, with n the
        number of nodes). If not None, use the distances given in the matrix to
        compute the distance from the root.
        """
        degrees = get_degree(self.graph.minimum_spanning_tree)
        roots = set(numpy.where(degrees == 1)[0])
        root = numpy.random.choice(list(roots))
        bfs = self.find_subtree(root, pdist=pdist)
        diameter = bfs[-1].distance
        return diameter

    def get_fork_depth(self, root):
        """
        Get the fork depth from the given root
        e.g.:
         /1-2-3-4-5-...
        0-1-2-3
         \\1-2-3-4-5-...
        For the graph above the depth of the fork is 3
        """
        degree = get_degree(self.graph.minimum_spanning_tree)[root]
        bfs_dist = self.get_bfs_distances(root)
        depth_list = sorted(bfs_dist.keys())
        i = 1
        depth = depth_list[i]
        while len(bfs_dist[depth]) == degree:
            i += 1
            depth = depth_list[i]
        return depth_list[i - 1]

    def get_fork_neighbors(self, fork):
        """
        Get all the fork neighbors from the given fork
        e.g.:
          8-9-10
         /        12-...
        0-1-2    /
        \\4-5-6-7-11-...
        The forks neighbors are 2, 7 and 10,
        with distances: 2, 4 and 3 respectively.
        """
        neighbors = self.G[fork].keys()
        fork_neighbors = []
        distances = []
        paths = []
        for neighbor in neighbors:
            vpath = self.get_vine_path(neighbor, exclusion_set=set(self.forks))
            if self.degrees[vpath[-1]] == 1:  # End of chain
                fork_neighbors.append(vpath[-1])
                paths.append(vpath)
                if self.density is None:
                    distances.append(len(vpath))
                else:
                    densities = self.density.get_density(self.coords[vpath])
                    distances.append(numpy.sum(densities))
            else:  # Node connected to a fork
                for node in self.G[vpath[-1]].keys():
                    if node != fork and node in self.forks:
                        fork_neighbors.append(node)
                        paths.append(vpath)
                        if self.density is None:
                            distances.append(len(vpath) + 1)
                        else:
                            densities = self.density.get_density(self.coords[vpath])
                            distances.append(numpy.sum(densities) +\
                                    self.density.get_density(self.coords[node]))
        return fork_neighbors, distances, paths

    def get_forks_graph(self):
        """
        Build the adjacency matrix of the new graph of forks F
        """
        F = {}  # format {i1:{j1:d1, j2:d2, ...}, ...}
        P = {}  # The vine paths from the forks
        for fork in self.forks:
            #print "Fork: %d/%d"%(i+1, len(self.forks))
            fork_neighbors, distances, paths = self.get_fork_neighbors(fork)
            F[fork] = dict(zip(fork_neighbors, distances))
            P[fork] = dict(zip(fork_neighbors, paths))
        self.F = F  # The Forks graph
        self.P = P
        return F

    @property
    def forks_score(self):
        """
        Get the score of each fork:
        The sum of the distances for each fork neighbors
        """
        if self.F is None:
            self.get_forks_graph()
        out = {}
        for fork in self.F:
            out[fork] = numpy.sum(list(self.F[fork].values())) / numpy.min(list(self.F[fork].values()))
        # dictionary sorted by value (first value highest score)
        out = collections.OrderedDict(sorted(out.items(), key=lambda t: t[1], reverse=True))
        return out

    def find_subtrees(self, return_forest=False):
        """
        Find all subtrees
        • If return_forest is True: return the bfs for each subtree (component)
        """
        degrees = get_degree(self.graph.minimum_spanning_tree)
        roots = set(numpy.where(degrees == 1)[0])
        limits = []
        visited_nodes = set()
        forest = []
        while len(roots) > 0:
            root = numpy.random.choice(list(roots))
            subtree = self.find_subtree(root)
            visited_nodes |= set([n.index for n in subtree])
            roots -= visited_nodes
            start, end = subtree[0].index, subtree[-1].index
            forest.append(subtree)
            limits.append((start, end))
        if return_forest:
            return limits, forest
        else:
            return limits

    def substract_from_tree(self, path, mutate=True):
        """
        Substract the nodes given in path from self.graph.minimum_spanning_tree
        • If mutate is True: replace self.graph.minimum_spanning_tree by the new
        minimum_spanning_tree
        • else: create a new object
        """
        row, col, data = [], [], []
        for i, j, value in zip(self.graph.minimum_spanning_tree.row, self.graph.minimum_spanning_tree.col,
                               self.graph.minimum_spanning_tree.data):
            if not i in path and not j in path:
                row.append(i)
                col.append(j)
                data.append(value)
        mstree = scipy.sparse.coo_matrix((data, (row, col)),
                                         shape=\
                                     self.graph.minimum_spanning_tree.shape)
        if mutate:
            self.graph.minimum_spanning_tree = mstree
            self.G = self.graph.get_graph(self.graph.minimum_spanning_tree)
            self.degrees = get_degree(self.graph.minimum_spanning_tree)
            self.n_nodes = self.graph.minimum_spanning_tree.shape[0]
            self.forks = list(numpy.where(self.degrees > 2)[0])
        else:
            return mstree

    def add_to_tree(self, mstree):
        """
        Add the mstree to self.graph.minimum_spanning_tree
        """
        out_mstree = self.graph.minimum_spanning_tree.tocsr()
        for i, j, value in zip(mstree.row, mstree.col, mstree.data):
            out_mstree[i, j] = value
        self.graph.minimum_spanning_tree = out_mstree.tocoo()
        self.G = self.graph.get_graph(self.graph.minimum_spanning_tree)

    def get_paths(self):
        """
        Get all the simple paths
        """
        paths = []
        while (1 - numpy.isinf(self.graph.minimum_spanning_tree)).sum() > 0:
            degrees = (1 - numpy.isinf(self.graph.minimum_spanning_tree)).sum(axis=0)
            roots = set(numpy.where(degrees == 1)[0])
            root = numpy.random.choice(list(roots))
            bfs = self.find_subtree(root)
            start, end = bfs[0].index, bfs[-1].index
            path = self.graph.shortestPath(start, end)
            self.substract_from_tree(path)
            paths.append(path)
        return paths

    def get_vine_path(self, node, exclusion_set=set(), coords=None, start=None, end=None):
        """
        Get the vine path for the given node. Node should be of degree 1
        • exclusion_set: set of nodes to exclude from the path
        • If coords is not None return the distance in the coordinates space
        • If start and end are not None, only the sub vine path between start
        and end is returned
        """
        path = []
        path.append(node)
        if coords is not None:
            distance = 0
        if node in self.G:
            while len(set(self.G[node].keys()) - exclusion_set) <= 2:
                neighbors = set(self.G[node].keys())
                neighbors -= exclusion_set
                neighbors -= set(path)
                if len(neighbors) == 0:
                    break
                for n in neighbors:
                    if coords is not None:
                        distance += numpy.linalg.norm(coords[n] - coords[path[-1]])
                    path.append(n)
                    node = n
            if start is not None and end is not None and\
                start in path and end in path:
                subpath = []
                addtopath = False
                order = 1
                if path.index(start) > path.index(end):
                    start, end = end, start
                    order = -1
                for e in path:
                    if e == start:
                        addtopath = True
                    if addtopath:
                        subpath.append(e)
                    if e == end:
                        addtopath = False
                        break
                path = subpath[::order]
        if coords is not None and start is None:
            return path, distance
        else:
            return path

    def get_vine_paths(self):
        """
        Get all the possible vine paths.
        Return: a dictionnary of vine paths:
        {n_i:[[v1,v2,v3,...], [w1,w2,...]], n_j:[[...]], ...}:
        the vine path ending at node n_i
        """
        degrees = get_degree(self.graph.minimum_spanning_tree)
        roots = numpy.where(degrees == 1)[0]
        vine_paths = {}
        for root in roots:
            vpath = self.get_vine_path(root)
            key = vpath[-1]
            if key not in vine_paths:
                vine_paths[key] = [
                    vpath[:-1],
                ]
            else:
                vine_paths[key].append(vpath[:-1])
        return vine_paths

    def path_cover_algorithm(self):
        """
        Path Cover Algorithm from OPTIMAL HAMILTONIAN COMPLETIONS AND PATH 
        COVERS FOR TREES, AND A REDUCTION TO MAXIMUM FLOW
        """
        path_cover = numpy.ones_like(self.graph.minimum_spanning_tree) * numpy.inf
        while (1 - numpy.isinf(self.graph.minimum_spanning_tree)).sum() > 0:
            #for i in range(10):
            nodes = numpy.asarray(self.find_subtrees()).flatten()
            node = numpy.random.choice(nodes)
            vine_path = self.get_vine_path(node)
            # fill the path_cover adjacency matrix
            for n1, n2 in zip(vine_path, vine_path[1:]):
                path_cover[n1, n2] = self.graph.minimum_spanning_tree[n1, n2]
                path_cover[n2, n1] = self.graph.minimum_spanning_tree[n2, n1]
            self.substract_from_tree(vine_path)
        self.graph.minimum_spanning_tree = path_cover

    def union(self, nodes):
        """
        Get the tree with the given nodes in respect with the topology of
        self.graph.minimum_spanning_tree, which must be a vine tree.
        E.G.:
        self.graph.minimum_spanning_tree: A-B-C-D-E-F-G-H
        nodes: [B, H, A, F]
        union(nodes) -> A-B-F-H
        """
        start, _ = self.find_subtrees()[0]
        vine_path = self.get_vine_path(start)  # vine path
        row, col, data = [], [], []
        pointer = -1
        for node in vine_path:
            if node in nodes:
                pointer += 1
                if pointer > 0:
                    row.append(node_prev)
                    col.append(node)
                    data.append(1.)
                    row.append(node)
                    col.append(node_prev)
                    data.append(1.)
                node_prev = node
        adjmat = scipy.sparse.coo_matrix((data, (row, col)),\
                                   shape=self.graph.minimum_spanning_tree.shape)
        return adjmat

    def resample(self, delta=3.8):
        """
        Resample the tree to get evenly spaced nodes
        • delta: distance between beads in Angstrom (default: 3.8 A -- average
        distance between consecutive C-alpha)
        """
        self.get_forks_graph()
        fragments = []
        fragment_ids = []
        fork_ids = []
        for fork in self.P:
            for end_node in self.P[fork]:
                fragment = []
                fragment.append(fork)
                fragment.extend(self.P[fork][end_node])
                if end_node not in fragment:
                    fragment.append(end_node)
                if tuple(fragment) not in fragment_ids and tuple(fragment[::-1]) not in fragment_ids and len(fragment) > 5:
                    fragment_ids.append(tuple(fragment))
                    fragments.append(resample_path(self.coords[fragment], delta))
        fragments = clean_path(fragments, self.density.x_step)
        return fragments
