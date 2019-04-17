# Phantom (i.e. 3D subject model with T1,T2,PD,df maps) functionalities

import numpy as np
import scipy.signal as ss

class Phantom:
    def __init__(self,T1map,T2map,PDmap,vsize,dBmap=0,loc=(0,0,0)):
        self.T1map = T1map
        self.T2map = T2map
        self.PDmap = PDmap
        self.vsize = vsize
        self.dBmap = dBmap
        self.loc = loc

        # Find field-of-view
        self.fov = vsize*np.array(np.shape(T1map))

        # Make location vectors
        ph_shape = np.shape(self.PDmap)

        # Define coordinates
        self.Xs = self.loc[0]+np.arange(-self.fov[0] / 2 + vsize / 2, self.fov[0] / 2, vsize)
        self.Ys = self.loc[1]+np.arange(-self.fov[1] / 2 + vsize / 2, self.fov[1] / 2, vsize)
        self.Zs = self.loc[2]+np.arange(-self.fov[2] / 2 + vsize / 2, self.fov[2] / 2, vsize)

    def get_location(self,inds):
        return self.Xs[inds[0]], self.Ys[inds[1]], self.Zs[inds[2]]

    def get_fov(self):
        return self.fov

    def get_vsize(self):
        return self.vsize

    def get_T1map(self):
        return self.T1map

    def get_T2map(self):
        return self.T2map

    def get_PDmap(self):
        return self.PDmap

    def get_shape(self):
        return np.shape(self.PDmap)

    def get_params(self,indx):
        return self.PDmap[indx],self.T1map[indx],self.T2map[indx]

    def get_list_locs(self):
        list_locs = []
        for x in self.Xs:
            for y in self.Ys:
                for z in self.Zs:
                    list_locs.append((x, y, z))
        return list_locs

    def get_list_inds(self):
        list_inds = []
        sh = self.get_shape()
        for u in range(sh[0]):
            for v in range(sh[1]):
                for w in range(sh[2]):
                    list_inds.append((u,v,w))
        return list_inds


class DTTPhantom(Phantom):
    """
    Discrete tissue type phantom
    """
    def __init__(self,type_map,type_params,vsize,dBmap=0,loc=(0,0,0)):
        """
        Makes a discrete-tissue type phantom
        :param type_map: 3D array of natural numbers
        :param type_params: dict. (type number)->(PD,T1,T2)
        :param dBmap: 3D array of dB s
        """
        self._type_map = type_map
        self._type_params = type_params
        T1map = np.ones(np.shape(type_map))
        T2map = np.ones(np.shape(type_map))
        PDmap = np.zeros(np.shape(type_map))

        for x in range(np.shape(type_map)[0]):
            for y in range(np.shape(type_map)[1]):
                for z in range(np.shape(type_map)[2]):
                    PDmap[x,y,z] = type_params[type_map[x,y,z]][0]
                    T1map[x,y,z] = type_params[type_map[x,y,z]][1]
                    T2map[x,y,z] = type_params[type_map[x,y,z]][2]


        super().__init__(T1map,T2map,PDmap,vsize,dBmap,loc)


class BrainwebPhantom(Phantom):
    def __init__(self, filename,dsf=1,make2d=False,loc=0,dir='z',dBmap=0):
        dsf = int(np.absolute(dsf))
        bw_data = np.load(filename).all()
        params = {k: np.array([v[3],v[0],v[1]]) for k, v in bw_data['params'].items()}

        typemap =  bw_data['typemap']

        dr = 1e-3 # 1mm voxel size

        # If we want planar phantom, then let's take the slice!
        if make2d:
            if dir in ['sagittal','x']:
                n = np.shape(typemap)[0]
                xx = dr*(n-1)
                loc_ind = int((n/xx)*loc + n/2)
                if loc_ind < 0:
                    loc_ind = 0
                if loc_ind > n-1:
                    loc_ind = n-1
                typemap = typemap[[loc_ind],:,:]

            elif dir in ['coronal','y']:
                n = np.shape(typemap)[1]
                yy = dr*(n-1)
                loc_ind = int((n/yy)*loc + n/2)
                if loc_ind < 0:
                    loc_ind = 0
                if loc_ind > n - 1:
                    loc_ind = n - 1
                typemap = typemap[:,[loc_ind],:]

            elif dir in ['axial','z']:
                n = np.shape(typemap)[2]
                zz = dr*(n-1)
                loc_ind = int((n/zz)*loc + n/2)
                if loc_ind < 0:
                    loc_ind = 0
                if loc_ind > n - 1:
                    loc_ind = n - 1
                typemap = typemap[:,:,[loc_ind]]

        # Make parm maps from typemap
        a,b,c = np.shape(typemap)
        T1map = np.ones((a,b,c))
        T2map = np.ones((a,b,c))
        PDmap = np.zeros((a,b,c))

        for x in range(a):
            for y in range(b):
                for z in range(c):
                    PDmap[x,y,z] = params[typemap[x,y,z]][0]
                    T1map[x,y,z] = params[typemap[x,y,z]][1]
                    T2map[x,y,z] = params[typemap[x,y,z]][2]

        # Downsample maps
        a,b,c = np.shape(PDmap)

        if a == 1:
            ax = [1,2]
        elif b == 1:
            ax = [0,2]
        elif c == 1:
            ax = [0,1]
        else:
            ax = [0,1,2]

        for v in range(len(ax)):
            PDmap = ss.decimate(PDmap, dsf, axis=ax[v], ftype='fir')
            T1map = ss.decimate(T1map, dsf, axis=ax[v], ftype='fir')
            T2map = ss.decimate(T2map, dsf, axis=ax[v], ftype='fir')


        dr = dr*dsf
        PDmap = np.clip(PDmap,a_min=0,a_max=1)
        T1map = np.clip(T1map,a_min=0,a_max=None)
        T2map = np.clip(T2map,a_min=0,a_max=None)

        super().__init__(T1map,T2map,PDmap,dr,dBmap)



def makeSphericalPhantom(n,fov,T1s,T2s,PDs,radii,loc=(0,0,0)):
    """
    Make a simple spherical phantom with layers
    """
    radii = np.sort(radii)
    m = np.shape(radii)[0]
    vsize = fov/n
    type_map = np.zeros((n,n,n))
    type_params = {}
    for x in range(n):
        for y in range(n):
            for z in range(n):
                d = vsize*np.linalg.norm(np.array([x,y,z])-(n-1)/2)
                for k in range(m):
                    if d <= radii[k]:
                        type_map[x,y,z] = k+1
                        break

    type_params[0] = (0,1,1)
    for k in range(m):
        type_params[k+1] = (PDs[k],T1s[k],T2s[k])

    return DTTPhantom(type_map,type_params,vsize,loc)


def makePlanarPhantom(n,fov,T1s,T2s,PDs,radii,dir='z',loc=(0,0,0)):
    radii = np.sort(radii)
    m = np.shape(radii)[0]
    vsize = fov / n
    type_map = np.zeros((n, n, 1))
    type_params = {}
    for x in range(n):
        for y in range(n):
                d = vsize * np.linalg.norm(np.array([x, y]) - (n - 1) / 2)
                for k in range(m):
                    if d <= radii[k]:
                        type_map[x,y,0] = k + 1
                        break

    type_params[0] = (0, 1, 1)
    for k in range(m):
        type_params[k + 1] = (PDs[k], T1s[k], T2s[k])

    if dir == 'x':
        type_map = np.swapaxes(type_map,1,2)
        type_map = np.swapaxes(type_map,0,1)
    elif dir =='y':
        type_map = np.swapaxes(type_map,0,2)
        type_map = np.swapaxes(type_map,0,1)

    return DTTPhantom(type_map, type_params, vsize, loc)




# Default cylindrical Phantom
def makeCylindricalPhantom(dim=3,dir='z',n=16,dr=(0,0,0)): # TODO new phantom in the works
    type_map = {}
    type_params = {}
    R = 0.12 # m
    r = 0.03 # m
    h = 0.24 # m
    s2 = np.sqrt(2)
    s3 = np.sqrt(3)
    centers = [(0,R/2,0.08),(-R*s3/4,-R/4,0.08),(R*s3/4,-R/4,0.08), # PD spheres
               (R/(2*s2),R/(2*s2),0),(-R/(2*s2),R/(2*s2),0),(-R/(2*s2),-R/(2*s2),0),(R/(2*s2),-R/(2*s2),0), # T1 spheres
               (0,R/2,-0.08),(-R/2,0,-0.08),(0,-R/2,-0.08),(R/2,0,-0.08)] # T1 spheres
   # type_params = {0:(0.25,0.5,0.01),1:(1,,0.5),2:(1,,0.5),3:(1,,0.5),
    #            4:(1,2,0.5),5:(1,2,0.5),6:(1,,),7:(1,,0.5),
     #           8:(),9:(),10:(),11:()}



    if dim == 3:
        # 3D phantom
        phantom = 0
        for x in range(n):
            for y in range(n):
                for z in range(n):
  #                  d = vsize * np.linalg.norm(np.array([x, y, z]) - (n - 1) / 2)
                    for c in range(len(centers)):
   #                     if d <= radii[k]:
    #                        type_map[x, y, z] = k + 1
                            break
                    type_map[x,y,z] = 0

    elif dim == 2:
        # 2D phantom
        if dir == 'z':
            a = 0
        elif dir == 'x':
            a = 1
        elif dir == 'y':
            a = 2

#        phantom = Phantom(type_map,type_params,vsize,loc)

    else:
        raise ValueError('#Dimensions must be 2 or 3')
    return phantom


