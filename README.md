# EMEC: cryo-Electron Microscopy and Evolutionary Coupling

## Introduction
Electron cryo-microscopy (cryo-EM) has emerged as a powerful method to obtain three-dimensional (3D) structures of macromolecular complexes at atomic or near-atomic resolution. However, *de novo* building of atomic models from near-atomic resolution (3-5 Å) cryo-EM density maps is a challenging task, in particular since poorly resolved side-chain densities hamper sequence assignment by automatic procedures at a lower resolution. Furthermore, segmentation of EM density maps into individual subunits remains a difficult problem when the structure of the subunits is not known, or when significant conformational rearrangement occurs between the isolated and associated form of the subunits. To tackle these issues, we have developed a graph-based method to thread most of the C-alpha trace of the protein backbone into the EM density map. The EM density is described as a weighted graph such that the resulting minimum spanning tree encompasses the high-density regions of the map. A pruning algorithm cleans the tree and finds the most probable positions of the C-alpha atoms, using side-chain density when available, as a collection of C-alpha trace fragments. By complementing experimental EM maps with contact predictions from sequence co-evolutionary information, we demonstrate that this approach can correctly segment EM maps into individual subunits and assign amino acids sequence to backbone traces to generate atomic models.

---

## Requirements

The following python packages are required and can be installed using pip:

- numpy
- mrcfile
- netCDF4
- scipy
- networkx
- matplotlib
- biopython
- mpi4py

EMEC makes use of `map_align` C++ implementation. This program can be installed using detailed instruction from the `map_align` repository:

https://github.com/sokrypton/map_align

For the final model building, Modeller software is used. This program can be installed using the standard installation procedure as described in:

https://salilab.org/modeller/download_installation.html

## Installation

To install EMEC, just clone the current repository and add the files in a directory that is listed in your `PYTHONPATH` variable, or add the path to EMEC directory in the `PYTHONPATH` variable. See https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH for more details.

The file `emec.py` can be linked as `emec` in a directory listed in your `$PATH` variable. If `$HOME/bin` is listed in `$PATH`:

`cd $HOME/bin && ln -s path/to/emec.py emec`

## Usage

The main EMEC file is `emec.py`. This file read a configuration file called `emec.conf`. An example `emec.conf` file is in `run/emec.conf`.
To reproduce the results of the paper, just run the following command in the `run/` directory:

`emec emec.conf`

The EM map is read from a NetCDF file format that can be created from an MRC file using the UCSF Chimera software (https://www.cgl.ucsf.edu/chimera/)
