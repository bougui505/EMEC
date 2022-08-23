#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-01-17 13:04:41 (UTC+0100)

import netCDF4
import numpy
import scipy.spatial

def interpolate(a, b, n):
    """
    linear interpolation of size n between two points a and b
    """
    v = b - a
    return numpy.asarray([a + v*i/(n-1) for i in range(n)])

class Density(object):
    """
    Object to read and work from em density
    """
    def __init__(self, emd, threshold):
        self.density, (self.xgrid, self.ygrid, self.zgrid), self.nx, self.ny, self.nz, self.x_step, self.y_step, self.z_step = self.read_emd(emd, -numpy.inf)
        self.density[self.density < threshold] = 0.
        self.density = self.normalize()
        self.kdtree = scipy.spatial.cKDTree(numpy.c_[self.xgrid, self.ygrid, self.zgrid])
        self.step = numpy.linalg.norm([self.x_step, self.y_step, self.z_step])

    def read_emd(self, emd, threshold):
        """
        Read the netcdf emd file with the given threshold
        Returns:
        • density: the electron density for the given threshold (dimension: (n),
          with n the number of grid points)
        • mgrid: the coordinates of the grid points (dimension: (n,3))
        """
        emd = netCDF4.Dataset(emd, mode='r')
        x_ori, y_ori, z_ori = emd.xyz_origin
        x_step, y_step, z_step = emd.xyz_step
        density = emd.variables['data'][:].T # cause netcdf return z,y,x
        nx, ny, nz = density.shape
        x_axis = numpy.linspace(x_ori, x_ori+nx*x_step, nx+1)[:-1]
        y_axis = numpy.linspace(y_ori, y_ori+ny*y_step, ny+1)[:-1]
        z_axis = numpy.linspace(z_ori, z_ori+nz*z_step, nz+1)[:-1]
        xgrid, ygrid, zgrid = numpy.asarray(numpy.meshgrid(x_axis,
                                             y_axis, z_axis, indexing='ij'))
        selection = density > threshold
        density = density[selection]
        xgrid, ygrid, zgrid = xgrid[selection], ygrid[selection], zgrid[selection]
        return density, (xgrid, ygrid, zgrid), nx, ny, nz, x_step, y_step, z_step

    def normalize(self):
        """
        Convert the density to a probability distribution that can be used by
        numpy.random.choice
        """
        p = self.density
        p -= p.min()
        p /= p.max()
        return p

    def digitize(self, coords):
        """
        Home made implementation of digitize:
        Return the index in the density for the given coordinates
        """
        return self.kdtree.query(coords)[1]

    def get_density(self, coords):
        """
        Return the density for the given coordinates
        """
        bins = self.digitize(coords)
        return self.density[bins]

    def get_edge_density(self, n1, n2):
        """
        return the minimal density along a given edge between 
        the two nodes n1-n2
        • n1, n2: the 3D coordinates of the 2 nodes
        """
        npoint = int(numpy.linalg.norm(n2 - n1) * 100/3.8)
        edge_points = interpolate(n1, n2, npoint)
        w = self.get_density(edge_points).min()
        return w

    def get_fitness_score(self, coords):
        """
        return a score to assess the quality of the fit to the EM density for
        the given coordinates
        """
        fitness = []
        for b1, b2 in zip(coords, coords[1:]):
            fitness.append(self.get_edge_density(b1, b2))
        return -numpy.log(min(fitness))
