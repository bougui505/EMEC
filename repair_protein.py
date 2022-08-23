#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2016-09-09 10:57:07 (UTC+0200)

import sys
import modeller
from modeller.automodel import autosched, refine
from modeller.optimizers import actions
from modeller import physical
import os
import numpy


def align_structures(pdb1, pdb2):
    """
    Align pdb2 on pdb1 and save it in outfilename
    """
    env = modeller.environ()
    mdl1 = modeller.model(env)
    mdl2 = modeller.model(env)
    mdl1.read(pdb1)
    mdl2.read(pdb2)
    aln = modeller.alignment(env)
    aln.append_model(mdl1, pdb1)
    aln.append_model(mdl2, pdb2)
    aln.malign3d(write_fit=True, write_whole_pdb=True, edit_file_ext=('.pdb', '.pdb'))


def repair_protein(pdbfile,
                   outpdb="modeller_out.pdb",
                   sequence=None,
                   write_restraint_file=False,
                   restraint_file=None,
                   extra_restraint_file=None,
                   flex=None,
                   deviation=999.,
                   write_psf=False,
                   emd=None,
                   resolution=None,
                   alpha=None,
                   strand=None):
    """

    If sequence is not None, the sequence is read and align to the sequence of
    the pdb file to add extra missing residues to the pdb file.

    If restraint_file is not None the restraints are read from the file instead
    of being computed.

    If extra_restraint_file is not None, the restraints are read from the given
    file and added to the set of computed restraints

    flex: selected residues to be in a loop (add flexibility to that
    bunch of residues). Syntax: [[5,10], [20,30]] for residues 5 to 10 and 20 to
    30.

    deviation: perturbation added in Angstrom for flexible parts.

    • emd: EM density file in MRC format
    • resolution: resolution of the EM density file
    • alpha: This makes restraints enforcing an α-helix
             (mainchain conformation class “A”) for the residue segment
             specified.
             (e.g.: [(8, 13), (19, 25), (28, 32), (54, 60), (63, 75)])
    • strand: This makes restraints enforcing an extended β-strand conformation
              for the residue segment specified by residues.
              (e.g.: [(8, 13), (19, 25), (28, 32), (54, 60), (63, 75)])

    """
    env = modeller.environ(rand_seed=-numpy.random.randint(low=2, high=50000))
    mdl = modeller.model(env)
    mdl.read(pdbfile)
    aln = modeller.alignment(env)

    a = mdl.atoms
    n_input_atoms = mdl.natm
    print("Number of input atoms: %d" % n_input_atoms)

    seq = []
    resid = int(mdl.residues[0].num)
    for res in mdl.residues:
        if int(res.num) - resid > 1:  # If not consecutive resids add chain break
            seq.append('/')
        resid = int(res.num)
        seq.append(res.code)
    seq = ''.join(seq)
    aln.append_model(mdl, 'input')
    print(seq)

    # reading the density
    if emd is not None:
        print("Reading density file")
        den = modeller.density(env, file=emd, em_density_format='MRC', density_type='GAUSS', resolution=resolution)

        env.edat.density = den
        env.edat.dynamic_sphere = True


# Read parameters (needed to build models from internal coordinates)
    env.libs.topology.read('${LIB}/top_heav.lib')
    env.libs.parameters.read('${LIB}/par.lib')

    mdl = None
    mdl = modeller.model(env)

    # Build a full atom model based on the sequence
    if sequence is not None:
        # Read the sequence from the fasta file
        data = numpy.genfromtxt(sequence, skip_header=1, dtype=str)
        if numpy.size(data.shape) > 0:  # multiple lines
            seq = ''.join(data)
        else:
            seq = str(data)
        aln.append_sequence(seq)
        aln.align()
        try:
            aln.write(file=sys.stdout)
        except TypeError:
            # If no stdout
            pass
        mdl.clear_topology()
        mdl.generate_topology(aln[""])
    else:
        mdl.build_sequence(seq)
        aln.append_model(mdl, 'output')
    mdl.transfer_xyz(aln)
    mdl.build(build_method='INTERNAL_COORDINATES', initialize_xyz=False)
    loops = mdl.loops(aln, 0, 9999, 0, 0)
    loop_selection = modeller.selection()
    for loop in loops:
        loop_selection.add(loop)
    if flex is not None:
        for sel_ in flex:
            loop_selection.add(mdl.residue_range('%s' % sel_[0], '%s' % sel_[1]))
    if len(loops) > 0:
        loop_selection.randomize_xyz(deviation=deviation)  # Add a random number
        #uniformly distributed in the interval from -deviation to
        #+deviation angstroms
    #set up restraints before computing energy
    rsr = mdl.restraints
    sel = modeller.selection(mdl)

    if restraint_file is not None:
        rsr.append(restraint_file)
    else:
        # MAKE DISTANCE RESTRAINTS
        # From /usr/lib/modeller9.16/modlib/modeller/automodel/automodel.py
        # (distance_restraints method)
        # Only do the standard residue types for CA, N, O, MNCH, SDCH dst rsrs
        # (no HET or BLK residue types):
        stdres = sel.only_std_residues() - loop_selection
        calpha = stdres.only_atom_types('CA')
        nitrogen = stdres.only_atom_types('N')
        oxygen = stdres.only_atom_types('O')
        mainchain = stdres.only_mainchain()
        sidechain = stdres - mainchain
        max_ca_ca_distance = 14.0
        max_n_o_distance = 11.0
        max_sc_mc_distance = 5.5
        max_sc_sc_distance = 5.0
        spline_on_site = True
        for (dmodel, maxdis, rsrrng, rsrsgn, rsrgrp, sel1, sel2, stdev) in \
            ((5, max_ca_ca_distance, (2, 99999), True,
              physical.ca_distance, calpha, calpha, (0, 1.0)),
             (6, max_n_o_distance, (2, 99999), False,
              physical.n_o_distance, nitrogen, oxygen, (0, 1.0)),
             (6, max_sc_mc_distance, (1, 2), False,
              physical.sd_mn_distance, sidechain, mainchain, (0.5, 1.5)),
             (6, max_sc_sc_distance, (2, 99999), True,
              physical.sd_sd_distance, sidechain, sidechain, (0.5, 2.0))):
            if len(sel1) > 0 and len(sel2) > 0:
                rsr.make_distance(sel1,
                                  sel2,
                                  aln=aln,
                                  spline_on_site=spline_on_site,
                                  distance_rsr_model=dmodel,
                                  restraint_group=rsrgrp,
                                  maximal_distance=maxdis,
                                  residue_span_range=rsrrng,
                                  residue_span_sign=rsrsgn,
                                  restraint_stdev=stdev,
                                  spline_range=4.0,
                                  spline_dx=0.7,
                                  spline_min_points=5)
        if extra_restraint_file is not None:
            rsr.append(extra_restraint_file)
        rsr.condense()
        if write_restraint_file:
            rsr.write('%s_dist_restraints.rsr' % os.path.splitext(outpdb)[0])

    for typ in ('stereo', 'phi-psi_binormal'):
        rsr.make(sel, restraint_type=typ, aln=aln, spline_on_site=True)
    for typ in ('omega', 'chi1', 'chi2', 'chi3', 'chi4'):
        rsr.make(sel,
                 restraint_type=typ + '_dihedral',
                 spline_range=4.0,
                 spline_dx=0.3,
                 spline_min_points=5,
                 aln=aln,
                 spline_on_site=True)
    if alpha is not None:
        # Add alpha helix restraints
        for res_start, res_end in alpha:
            rsr.add(modeller.secondary_structure.alpha(\
                    mdl.residue_range('%d:'%res_start, '%d:'%res_end)))
    if strand is not None:
        # Add beta strand restraints
        for res_start, res_end in strand:
            rsr.add(modeller.secondary_structure.strand(\
                    mdl.residue_range('%d:'%res_start, '%d:'%res_end)))
    rsr.condense()
    if write_restraint_file:
        rsr.write('%s_topology.rsr' % os.path.splitext(outpdb)[0])

    libsched = autosched.normal
    mysched = libsched.make_for_model(mdl) * env.schedule_scale
    # Optimize for all steps in the schedule
    for step in mysched:
        step.optimize(sel, output='REPORT', max_iterations=200)
    mdl.optimize_output = 'REPORT'
    refine.very_fast(sel, actions=actions.trace(10))
    mdl.res_num_from(modeller.model(env, file=pdbfile), aln)
    mdl.write(outpdb)
    #align_structures(pdbfile, outpdb)
    if write_psf:
        basename = os.path.splitext(outpdb)[0]
        mdl.write_psf('%s.psf' % basename)

if __name__ == '__main__':
    pdbfile = sys.argv[1]
    if len(sys.argv) == 3:
        sequence = sys.argv[2]
    else:
        sequence = None
    repair_protein(pdbfile, sequence=sequence, outpdb="%s_fixed.pdb" % os.path.splitext(pdbfile)[0])
