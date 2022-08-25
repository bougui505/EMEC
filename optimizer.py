#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-01-04 14:14:39 (UTC+0100)

import numpy
import scipy.optimize
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
import EMDensity


def build_database():
    """
    Build the database on pseudo dihedral for C alpha trace
    Build the databse:
    {(b1,b2):{[s1,s2,s3,s4,s5]:n}}
    b: angles
    s: secondary structure assignment
    n: count
    Other entries:
    total: Total number of count
    ss: Total number of this secondary structure assignment
    ss could be for example ('-', 'E', '-', 'H', 'H')
    """
    table = numpy.genfromtxt("CADihedralRestraint.dat", dtype=str)
    database = {}
    total = 0
    for line in table:
        angles = tuple(numpy.int_(line[1:3]))
        ss = line[0]
        ss = tuple([e for e in ss])
        n = int(line[-1])
        total += n
        if angles not in database:
            database[angles] = {ss: n}
        else:
            database[angles][ss] = n
        if ss not in database:
            database[ss] = n
        else:
            database[ss] += n
    database['total'] = total
    return database


def get_pseudo_ramachandran(dihedral_database):
    """
    Compute the pseudo ramachandran
    """
    rama = numpy.zeros((36, 36))
    # ramachandran for helix (H), beta strand (E) and coil (_)
    rama_H = numpy.zeros((36, 36))
    rama_E = numpy.zeros((36, 36))
    rama__ = numpy.zeros((36, 36))
    for i, d1 in enumerate(range(0, 360, 10)):
        for j, d2 in enumerate(range(0, 360, 10)):
            mapping = dihedral_database[d1, d2]
            count = numpy.sum(mapping.values())
            rama[i, j] = count
            if ('H', ) * 5 in mapping:
                rama_H[i, j] = mapping[('H', ) * 5]
            if ('E', ) * 5 in mapping:
                rama_E[i, j] = mapping[('E', ) * 5]
            if ('-', ) * 5 in mapping:
                rama__[i, j] = mapping[('-', ) * 5]
    rama /= rama.max()
    rama_H /= rama_H.max()
    rama_E /= rama_E.max()
    rama__ /= rama__.max()
    return rama, rama_H, rama_E, rama__


def get_ca_angle_potential(ca_angle_database='CAAngleRestraint.dat'):
    """
    Return a function that compute the statistical potential for the given angle
    between 3 CA.
    The function returned is defined on the [0,180] interval.
    """
    ca_angle_potential = numpy.genfromtxt(ca_angle_database)[:, 1:3]
    bins = {}
    for l in ca_angle_potential:
        alpha, count = l
        if alpha not in bins:
            bins[alpha] = count
        else:
            bins[alpha] += count
    ca_angle_potential = numpy.asarray(bins.items())
    ca_angle_potential = numpy.insert(ca_angle_potential, 0, [0, 0], axis=0)
    ca_angle_potential = numpy.append(ca_angle_potential, [[180, 0]], axis=0)
    ca_angle_potential[:, 1] /= ca_angle_potential[:, 1].max()
    f = interp1d(ca_angle_potential[:, 0], ca_angle_potential[:, 1], kind='linear')
    return f


def get_dihedral(coords):
    """
    Dihedral angle between 3 vectors.
    • coords: coords of 4 points that define the 3 vectors:
    np.array of shape 4*3
    See:
    http://math.stackexchange.com/a/47084/192193
    """
    p1, p2, p3, p4 = coords
    v1 = (p2 - p1)
    v1 /= numpy.linalg.norm(v1)
    v2 = (p3 - p2)
    v2 /= numpy.linalg.norm(v2)
    v3 = (p4 - p3)
    v3 /= numpy.linalg.norm(v3)
    n1 = numpy.cross(v1, v2)
    n2 = numpy.cross(v2, v3)
    m1 = numpy.cross(n1, v2)
    x = n1.dot(n2)
    y = m1.dot(n2)
    dihedral = numpy.arctan2(y, x)
    if dihedral > 0:
        return 2 * numpy.pi - numpy.arctan2(y, x)
    else:
        return -dihedral


def get_dihedrals(coords):
    """
    return the pseudo dihedrals (theta1, theta2) from coords
    """
    vectors = numpy.asarray(zip(coords, coords[1:], coords[2:], coords[3:]))
    dihedrals = numpy.asarray([get_dihedral(e) for e in vectors])
    dihedrals = numpy.rad2deg(dihedrals)
    dihedrals = numpy.asarray(zip(dihedrals, dihedrals[1:]))
    return dihedrals


def get_angles(coords):
    """
    return the angle between adjacent triplets of CA
    """
    bonds = numpy.diff(coords, axis=0)
    bonds /= numpy.linalg.norm(bonds, axis=1)[:, None]
    cos_val = bonds.dot(bonds.T).diagonal(offset=1)
    # clamp between -1 and 1
    cos_val = numpy.clip(cos_val, -1., 1.)
    angles = numpy.rad2deg(numpy.arccos(cos_val))
    angles = 180. - angles
    return angles


def get_angle_energies(coords_flatten, ca_angle_potential):
    """
    Compute the per residue decomposition bond energy
    """
    n = coords_flatten.size
    coords = coords_flatten.reshape(n / 3, 3)
    angles = get_angles(coords)
    return -numpy.log(ca_angle_potential(angles))


def get_angle_energy(coords_flatten, ca_angle_potential):
    """
    Compute the potential for a given angle between a triplet of CA
    """
    return get_angle_energies(coords_flatten, ca_angle_potential).sum()


def get_dihedral_energies(coords_flatten, rama):
    """
    Compute the pseudo dihedral energy per residue
    """
    n = coords_flatten.size
    coords = coords_flatten.reshape(n / 3, 3)
    dihedrals = get_dihedrals(coords)
    digit = numpy.asarray(
        zip(numpy.digitize(dihedrals[:, 0], range(0, 360, 10)), numpy.digitize(dihedrals[:, 1], range(0, 360, 10)))).T
    return -numpy.log(rama[tuple(digit - 1)])


def get_dihedral_energy(coords_flatten, rama):
    """
    Compute the total pseudo dihedral energy based on the pseudo ramachandran (rama)
    """
    return get_dihedral_energies(coords_flatten, rama).sum()


def get_bond_energies(coords_flatten, bond_potential):
    """
    Compute the per residue decomposition of the bond_energy
    """
    n = coords_flatten.size
    coords = coords_flatten.reshape(n / 3, 3)
    return -numpy.log(bond_potential(numpy.linalg.norm(numpy.diff(coords, axis=0), axis=1)))


def get_bond_energy(coords_flatten, bond_potential):
    """
    Compute the total pseudo bond energy for consecutive Calpha based on
    bond_potential (lambda function)
    """
    return get_bond_energies(coords_flatten, bond_potential).sum()


def get_density_energies(coords_flatten, Density):
    """
    Compute the per residue decomposition of the density_energy
    Density: density object
    """
    n = coords_flatten.size
    coords = coords_flatten.reshape(n / 3, 3)
    return -numpy.log(Density.density[Density.digitize(coords)]).sum()


def get_density_energy(coords_flatten, Density):
    """
    Compute the total density energy. Assess the quality of the fit to the
    EM density
    """
    return get_density_energies(coords_flatten, Density).sum()


def secondary_structure_attribution(prob_H, prob_E, prob__):
    """
    Return secondary structure attribution:
    [helix, beta, -]
    """
    helix_set = set(numpy.where(numpy.logical_and(prob_H > prob_E, prob_H > prob__))[0])
    beta_set = set(numpy.where(numpy.logical_and(prob_E > prob_H, prob_E > prob__))[0])
    __set = set(numpy.where(numpy.logical_and(prob__ > prob_H, prob__ > prob_E))[0])
    ss_attr = []
    for ss_set in [helix_set, beta_set, __set]:
        ss_ = []
        start = None
        for i in range(len(prob_H) + 1):
            if i in ss_set:
                if start is None:
                    start = i
            else:
                if start is not None:
                    end = i - 1
                    ss_.append((start, end))
                    start = None
        ss_attr.append(ss_)
    return ss_attr

def write_pdb(data, occupancy=None, outfilename="points.pdb",
              connect=None, sequence=None, atoms=None, chain_ids=None,
              resids=None, write_conect=True, write_UNK=True):
    """
    write data (coordinates of points (n*3)) in a pdb file
    • sequence: if not None use the sequence (list of 1 letter code) to write to
    the PDB
    • atoms: list of atom types
    • chain_ids: list of chain ids
    • resids: list of residue ids
    • write_conect: If True write the CONECT fields
    • write_UNK: Write UNKnown residues ('non attributed') to PDB file
    """
    one_to_three = {'R': 'ARG', 'H': 'HIS', 'K': 'LYS', 'D': 'ASP', 'E': 'GLU',
                    'S': 'SER', 'T': 'THR', 'N': 'ASN', 'Q': 'GLN', 'C':'CYS',
                    'G': 'GLY', 'P': 'PRO', 'A': 'ALA', 'V': 'VAL', 'I': 'ILE',
                    'L': 'LEU', 'M': 'MET', 'F': 'PHE', 'Y': 'TYR', 'W': 'TRP',
                    'X': 'UNK'}
    if sequence is not None:
        aa_list = [one_to_three[e] if len(e) == 1 else e for e in sequence]
    else:
        aa_list = ['GLY', ]*data.shape[0]
    if not write_UNK:
        # Remove UNK from the data
        aa_list = numpy.asarray(aa_list)
        selection = aa_list != 'UNK'
        data = data[selection]
        aa_list = aa_list[selection]
    if atoms is None:
        atoms = [
            'CA',
        ] * data.shape[0]
    if chain_ids is None:
        chain_ids = [
            'A',
        ] * data.shape[0]
    if resids is None:
        resids = numpy.arange(data.shape[0]) + 1
    with open(outfilename, 'w') as outfile:
        atom_id = 0
        for i, e in enumerate(data):
            x, y, z = e
            x, y, z = "%.3f" % x, "%.3f" % y, "%.3f" % z
            if occupancy is None:
                if connect is None:
                    aa = aa_list[i]
                    if aa is not 'UNK' or write_UNK:
                        atom = atoms[i]
                        chain_id = chain_ids[i]
                        resid = resids[i]
                        atom_id += 1
                        outfile.write("%-6s%5s %4s %3s %s%4s    %8s%8s%8s\n"%("ATOM", atom_id, atom, aa, chain_id, resid, x, y, z))
                else:
                    if i+1 in numpy.asarray(connect).flatten(): # the atom is connected to at least one atom
                        outfile.write("%-6s%5s %4s %3s %s%4s    %8s%8s%8s\n"%("ATOM", i+1, "QA", "DUM", "A", i+1, x, y, z))
            else:
                o = "%.2f"%occupancy[i]
                if connect is None:
                    outfile.write("%-6s%5s %4s %3s %s%4s    %8s%8s%8s%6s\n"%("ATOM", i+1, "QA", "DUM", "A", i+1, x, y, z, o))
                else:
                    if len(connect[i]) > 1:
                        outfile.write("%-6s%5s %4s %3s %s%4s    %8s%8s%8s\n"%("ATOM", i+1, "QA", "DUM", "A", i+1, x, y, z))
        if write_conect:
            if connect is None:
                connect = []
                n = len(data)
                for i in range(n-1):
                    if resids[i+1] - resids[i] <= 1:
                        connect.append([i+1, i+2])
            if connect is not None:
                for connect_ids in connect:
                    if len(connect_ids) > 1:
                        outfile.write("CONECT")
                        for a in connect_ids:
                            outfile.write("%5d" % a)
                        outfile.write("\n")


def write_minimum_spanning_tree(data, minimum_spanning_tree, outfile='minimum_spanning_tree.pdb'):
    """
    Write the minimum spanning tree to a pdb file
    """
    connect = zip(minimum_spanning_tree.row + 1, minimum_spanning_tree.col + 1)
    write_pdb(data, connect=connect, outfilename=outfile)


class OPTIMIZE(object):
    """
    Optimize the geometry of the C-alpha trace based on statistical potential on
    distances and pseudo dihedral angles
    """
    def __init__(self,
                 coords,
                 emd=None,
                 level=None,
                 database=None,
                 angle_potential=None,
                 k_dihedral=1.,
                 k_bond=1.,
                 k_angle=1.,
                 k_density=1.):
        """
        • coords: numpy array with the coordinates of the C-alpha trace
        • emd: Name of the file containing the EM density. If None, the density
        associated energy is not computed.
        • level: level associated to the EM density
        • k_dihedral, k_bond, k_angle, k_density: relative weights for each
        energy term
        """
        self.k_dihedral, self.k_bond, self.k_angle, self.k_density = k_dihedral,\
                                                                     k_bond,\
                                                                     k_angle,\
                                                                     k_density
        if database is None:
            self.dihedral_database = build_database()
        else:
            self.dihedral_database = database
        self.rama, self.rama_H, self.rama_E, self.rama__ = get_pseudo_ramachandran(self.dihedral_database)
        self.bond_potential = lambda x: numpy.exp(-(x - 3.8)**2 / (2 * 0.1**2))
        if angle_potential is None:
            self.angle_potential = get_ca_angle_potential()
        else:
            self.angle_potential = angle_potential
        self.coords = coords
        if emd is not None:
            self.Density = EMDensity.Density(emd, level)
        else:
            self.Density = None
        self.dihedrals = get_dihedrals(coords)
        self.prob_H, self.prob_E, self.prob__ = self.get_ss_profile()
        # per_residue_decomposition_energy
        self.dihedral_e = None
        self.bond_e = None
        self.angle_e = None
        self.energy_per_bead = None

    def get_per_residue_decomposition_energy(self):
        """
        Compute the per_residue_decomposition_energy
        """
        self.dihedral_e = get_dihedral_energies(self.coords.flatten(), self.rama)
        dihedral_e = list(self.dihedral_e)
        dihedral_e.append(self.dihedral_e[-1])
        dihedral_e.append(self.dihedral_e[-1])
        dihedral_e.insert(0, dihedral_e[0])
        dihedral_e.insert(0, dihedral_e[0])
        self.dihedral_e = dihedral_e

        self.bond_e = get_bond_energies(self.coords.flatten(), self.bond_potential)
        # Get one value per bead:
        bond_e = list(numpy.asarray(zip(self.bond_e, self.bond_e[1:])).mean(axis=1))
        bond_e.append(self.bond_e[-1])
        bond_e.insert(0, self.bond_e[0])
        self.bond_e = bond_e

        self.angle_e = get_angle_energies(self.coords.flatten(), self.angle_potential)
        angle_e = list(self.angle_e)
        angle_e.append(self.angle_e[-1])
        angle_e.insert(0, self.angle_e[0])
        self.angle_e = angle_e

        self.energy_per_bead = numpy.r_[[self.bond_e, self.angle_e, self.dihedral_e]].sum(axis=0)
        return self.energy_per_bead

    def get_total_energy(self, coords_flatten):
        """
        Compute the total energy with the given factors (k)
        """
        if self.k_dihedral > 0.:
            e_dihedral = self.k_dihedral * get_dihedral_energy(coords_flatten, self.rama)
        else:
            e_dihedral = 0.
        if self.k_bond > 0.:
            e_bond = self.k_bond * get_bond_energy(coords_flatten, self.bond_potential)
        else:
            e_bond = 0.
        if self.Density is not None:
            e_density = self.k_density * get_density_energy(coords_flatten, self.Density)
        else:
            e_density = 0.
        if self.k_angle > 0.:
            e_angle = self.k_angle * get_angle_energy(coords_flatten, self.angle_potential)
        else:
            e_angle = 0.
        #print e_dihedral, e_bond, e_density, e_angle
        return e_dihedral + e_bond + e_density + e_angle

    def minimize(self, maxiter=None):
        """
        Minimize the total energy
        """
        print("Initial total energy: %.2f" % self.get_total_energy(self.coords.flatten()))
        if maxiter is None:
            res = scipy.optimize.minimize(self.get_total_energy, self.coords.flatten(), method='Powell')
        else:
            res = scipy.optimize.minimize(self.get_total_energy,
                                          self.coords.flatten(),
                                          method='Powell',
                                          options={'maxiter': maxiter})
        n = res.x.size
        new_coords = res.x.reshape((n / 3, 3))
        self.coords = new_coords
        self.dihedrals = get_dihedrals(new_coords)
        energy = self.get_total_energy(self.coords.flatten())
        print("Final total energy: %.2f" % energy)
        self.prob_H, self.prob_E, self.prob__ = self.get_ss_profile()
        return energy

    def get_ss_profile(self):
        """
        Compute the secondary structure profile
        """
        prob_H = self.secondary_structure_probability('H')
        prob_E = self.secondary_structure_probability('E')
        prob__ = self.secondary_structure_probability('-')
        # Normalize the ss profile:
        k = numpy.c_[prob_H, prob_E, prob__].sum(axis=1)
        prob_H = prob_H / k
        prob_E = prob_E / k
        prob__ = prob__ / k
        return prob_H, prob_E, prob__

    def secondary_structure_probability(self, ss):
        """
        Compute the probability profile along the C alpha trace to be to the
        given secondary structure ss:
        ss= 'H' | 'E' | '-'
        """
        ss_keys = numpy.insert(numpy.asarray(list(itertools.product(['H', 'E', '-'], repeat=4))), 2, ss, axis=1)
        ss_keys = [tuple(e) for e in ss_keys]
        bins = numpy.arange(0, 360, 10)
        #ind = (numpy.digitize(self.dihedrals, bins) -1).flatten()
        # To be compatible with numpy version < 1.10.0
        ind = numpy.asarray(
            zip(numpy.digitize(self.dihedrals[:, 0], bins) - 1,
                numpy.digitize(self.dihedrals[:, 1], bins) - 1))
        #
        angle_keys = bins[ind]
        angle_keys = angle_keys.reshape(self.dihedrals.shape)
        angle_keys = [tuple(e) for e in angle_keys]
        probs = []
        for angle_key in angle_keys:
            count = 0
            tot = 0.
            for ss_key in ss_keys:
                if ss_key in self.dihedral_database[angle_key]:
                    count += self.dihedral_database[angle_key][ss_key]
                    tot += self.dihedral_database[ss_key]
            if tot > 0:
                probs.append(count / tot)
            else:
                probs.append(0)
        # Complete probabilities for the first two and last CA with a copy
        # of the probability of CA(2) and CA(n-2)
        probs = numpy.r_[[
            probs[0],
        ] * 2, probs, [
            probs[-1],
        ] * 2]
        return probs

    def plot_pseudo_ramachandran(self):
        """
        plot the pseudo ramachandran
        """
        plt.matshow(self.rama, extent=(0, 360, 360, 0), norm=LogNorm())
        plt.colorbar()
        plt.scatter(self.dihedrals[:, 0], self.dihedrals[:, 1], c='r')

    def plot_ss_profile(self):
        """
        plot the secondary structure profile
        """
        colors = ['r', 'orange', 'gray']
        plt.plot(self.prob_H, '.-', c=colors[0], label='H')
        plt.plot(self.prob_E, '.-', c=colors[1], label='E')
        plt.plot(self.prob__, '.-', c=colors[2], label='-')
        ss_attr = secondary_structure_attribution(self.prob_H, self.prob_E, self.prob__)
        for i, ss_ in enumerate(ss_attr):
            for se in ss_:
                plt.axvspan(se[0], se[1], alpha=0.25, color=colors[i])
        plt.ylabel('probability')
        plt.xlabel('sequence')
        plt.grid()
        plt.legend()
