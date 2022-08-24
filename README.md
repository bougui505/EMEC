# EMEC: cryo-Electron Microscopy and Evolutionary Coupling

## Introduction
Electron cryo-microscopy (cryo-EM) has emerged as a powerful method to obtain three-dimensional (3D) structures of macromolecular complexes at atomic or near-atomic resolution. However, *de novo* building of atomic models from near-atomic resolution (3-5 Å) cryo-EM density maps is a challenging task, in particular since poorly resolved side-chain densities hamper sequence assignment by automatic procedures at a lower resolution. Furthermore, segmentation of EM density maps into individual subunits remains a difficult problem when the structure of the subunits is not known, or when significant conformational rearrangement occurs between the isolated and associated form of the subunits. To tackle these issues, we have developed a graph-based method to thread most of the C-alpha trace of the protein backbone into the EM density map. The EM density is described as a weighted graph such that the resulting minimum spanning tree encompasses the high-density regions of the map. A pruning algorithm cleans the tree and finds the most probable positions of the C-alpha atoms, using side-chain density when available, as a collection of C-alpha trace fragments. By complementing experimental EM maps with contact predictions from sequence co-evolutionary information, we demonstrate that this approach can correctly segment EM maps into individual subunits and assign amino acids sequence to backbone traces to generate atomic models.

---

## Usage
