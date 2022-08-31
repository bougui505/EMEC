#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2016-12-19 09:28:25 (UTC+0100)

import numpy
from scipy.ndimage.morphology import distance_transform_edt
import scipy.spatial.distance
import scipy.stats
from EMEC import progress_reporting
import os
from EMEC import Graph
from EMEC import EMDensity
from EMEC import Tree


class Fit(object):
    """
    Fit C-alpha trace onto a electron microscopy density map
    """
    def __init__(self,
                 emd,
                 threshold,
                 graph,
                 coords,
                 n_iter,
                 alpha_0=1.,
                 alpha_1=0.01,
                 radius_0=5.,
                 radius_1=.1,
                 training_set=None,
                 refine=False,
                 refinment_radius=3.8 / 2.):
        """
        • emd: electron microscopy density in netcdf (nc) format
        • threshold: threshold applied to the em map
        • graph: graph object to fit to the density map
        • coords: coordinates for each node of the graph
        • n_iter: number of iterations
        • alpha_0 and alpha_1: starting and ending learning rate respectively
        • radius_0 and radius_1: starting and ending radius
        • training_set: training set to use. If None, the training set is 
        generated from the em density given in emd.
        • If refine is True, refine the trace locally in the density found
        around the trace (given by the refinment_radius).
        """
        self.graph = graph
        self.coords = coords
        self.compute_distances()
        self.Density = EMDensity.Density(emd, threshold)
        self.density = self.Density.density
        self.mgrid = (self.Density.xgrid, self.Density.ygrid, self.Density.zgrid)
        self.nx, self.ny, self.nz = self.Density.nx, self.Density.ny, self.Density.nz
        self.x_step, self.y_step, self.z_step = self.Density.x_step, self.Density.y_step, self.Density.z_step
        self.t = 0  # iteration index
        self.n_iter = n_iter
        if refine:
            selection = set(numpy.concatenate(self.Density.kdtree.query_ball_point(self.coords, refinment_radius)))
            n_voxels = self.Density.density.size
            self.Density.density[list(set(range(n_voxels)) - selection)] = 0.
            self.density = self.Density.density
        if training_set is None:
            self.training_set = self.generate_data_points()
        else:
            self.training_set = training_set
        self.annealing_radius = self.annealing_function(radius_0, radius_1)
        self.annealing_alpha = self.annealing_function(alpha_0, alpha_1)
        self.radius = self.annealing_radius(self.t)
        self.alpha = self.annealing_alpha(self.t)
        self.neighborhood_function = lambda x: self.alpha * numpy.exp(-x**2 / (2. * self.radius**2))
        self.bmu = None

    def compute_distances(self):
        """
        Replace the weights of the edges in self.graph.minimum_spanning_tree by
        the distances in the graph space
        """
        #pdist = scipy.spatial.distance.pdist(self.coords)
        #pdist = scipy.spatial.distance.squareform(pdist)
        for i, j in numpy.asarray(numpy.where(~numpy.isinf(self.graph.minimum_spanning_tree))).T:
            #self.graph.minimum_spanning_tree[i,j] = pdist[i, j]
            self.graph.minimum_spanning_tree[i, j] = 1.

    def generate_data_points(self):
        """
        Generate all data points from the probability density for training
        """
        index = numpy.random.choice(self.density.size, p=self.density / self.density.sum(), size=self.n_iter)
        xgrid, ygrid, zgrid = self.mgrid
        coord = numpy.asarray([xgrid[index], ygrid[index], zgrid[index]])
        return coord.T

    def get_learning_params(self, bmu):
        """
        Compute the learning parameters alpha*sigma:
        M(t+1) = M(t) + alpha*sigma*(V-M(t))
        """
        self.radius = self.annealing_radius(self.t)
        self.alpha = self.annealing_alpha(self.t)
        # The distance cutoff is 10 bonds
        distances = self.graph.dijkstra(bmu, max_distance=10.)
        alpha_sigma = self.neighborhood_function(distances)
        return alpha_sigma

    def annealing_function(self, param_0, param_1):
        """
        define a linear annealing function for the given parameters:
        • param_0: starting parameter
        • param_1: parameter reached for self.n_iter iterations
        """
        return lambda x: ((param_1 - param_0) / self.n_iter) * x + param_0

    def get_bmu(self, data):
        """
        Compute the Best Matching Unit (BMU) for the given data point
        """
        bmu = scipy.spatial.distance.cdist(data[None, :], self.coords).argmin()
        return bmu

    def get_bmus(self):
        """
        Compute the BMU of each data point in self.training_set
        """
        return scipy.spatial.distance.cdist(self.training_set, self.coords).argmin(axis=1)

    def apply_learning(self):
        """
        Apply the learning equation:
        M(t+1) = M(t) + alpha*sigma*(V-M(t))
        """
        data = self.training_set[self.t]
        self.bmu = self.get_bmu(data)
        learning_params = self.get_learning_params(self.bmu)
        #print self.coords.shape, learning_params.shape, (data - self.coords).shape
        self.coords += learning_params * (data - self.coords)
        self.t += 1

    def learn(self):
        """
        Learn the map for n_iter iterations
        """
        progress = progress_reporting.Progress(self.n_iter, delta=10, label="fitting")
        for i in range(self.n_iter):
            self.apply_learning()
            progress.count(report="iteration %d; α=%.2g; σ=%.2g; bmu=%d" % (i, self.alpha, self.radius, self.bmu))
