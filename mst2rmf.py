#!/home/bougui/source/imp/imp_release/setup_environment.sh python

def save_rmf(mst_dict,coords,rmffilename, densities=None):
    import IMP
    import IMP.atom
    import IMP.rmf
    import RMF
    mdl=IMP.Model()

    pnamedict={}
    p = IMP.Particle(mdl)
    hroot = IMP.atom.Hierarchy.setup_particle(p)
    hroot.set_name("root")

    keys=mst_dict.keys()
    for key in mst_dict:
        keys.extend(mst_dict[key].keys())

    keys=set(keys)

    for i, coord in enumerate(coords):
        if i not in keys: continue
        p = IMP.Particle(mdl)
        d=IMP.core.XYZR.setup_particle(p)
        d.set_coordinates(coord)
        d.set_radius(0.35)
        IMP.atom.Mass.setup_particle(p, 1.0)
        if densities is not None:
            dens=densities[i]/max(densities)
            c=IMP.display.Colored.setup_particle(p,IMP.display.get_rgb_color(dens))
        pnamedict[i]=p
        h = IMP.atom.Hierarchy.setup_particle(p)
        h.set_name(str(i))
        hroot.add_child(h)

    values = []
    for key in mst_dict:
        values.extend(mst_dict[key].values())
    max_val = max(values)

    sgs=[]
    for key1 in mst_dict:
        p1=pnamedict[key1]
        for key2 in mst_dict[key1]:
            p2=pnamedict[key2]
            w=mst_dict[key1][key2]
            c=IMP.display.get_rgb_color(w/max_val)
            #seg=IMP.algebra.Segment3D(IMP.core.XYZ(p1).get_coordinates(),IMP.core.XYZ(p2).get_coordinates())
            #sgs.append(IMP.display.SegmentGeometry(seg,c,p1.get_name()+"-"+p2.get_name()))

            if not IMP.atom.Bonded.get_is_setup(p1):
                IMP.atom.Bonded.setup_particle(p1)
            if not IMP.atom.Bonded.get_is_setup(p2):
                IMP.atom.Bonded.setup_particle(p2)
            if not IMP.atom.get_bond(IMP.atom.Bonded(p1),IMP.atom.Bonded(p2)):
                IMP.atom.create_bond(IMP.atom.Bonded(p1),IMP.atom.Bonded(p2),1)
    rh = RMF.create_rmf_file(rmffilename)
    IMP.rmf.add_hierarchies(rh, [hroot])
    #IMP.rmf.add_geometries(rh,sgs)
    #IMP.rmf.add_restraints(rh,[rslin])
    #IMP.rmf.add_geometries(rh, sgs)
    IMP.rmf.save_frame(rh)
    del rh


if __name__ == '__main__':
    #mst_dict={"A":{"B":0.5,"C":0.2},"B":{"C":0.9}}
    #coor_dict={"A":(0,0,0),"B":(0,0,1),"C":(0,1,0)}
    #save_rmf(mst_dict,coor_dict,"mst2rmf.rmf3")
    import sys
    import Graph
    import numpy
    import EMDensity
    mstree_npz = sys.argv[1]
    emd_nc = sys.argv[2]
    level = float(sys.argv[3])
    mstree = numpy.load(mstree_npz)
    coords = mstree['coords']
    mstree = mstree['mstree'].item()
    graph = Graph.Graph(mstree)

    emd = EMDensity.Density(emd_nc, level)
    densities = emd.get_density(coords)

    save_rmf(graph.get_graph(), coords, 'mstree.rmf', densities=densities)
