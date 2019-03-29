# Phantom (i.e. 3D subject model with T1,T2,PD,df maps) functionalities

import numpy as np


class Phantom:
    def __init__(self,T1map,T2map,PDmap,vsize,dBmap=0):
        self._T1map = T1map
        self._T2map = T2map
        self._PDmap = PDmap
        self._vsize = vsize
        self._dBmap = dBmap

        # Find field-of-view
        self._fov = vsize*np.array(np.shape(T1map))

        # Make location vectors
        ph_shape = np.shape(self._PDmap)

        # Define coordinates
        self._Xs = np.arange(-self._fov[0] / 2 + vsize / 2, self._fov[0] / 2, vsize)
        self._Ys = np.arange(-self._fov[1] / 2 + vsize / 2, self._fov[1] / 2, vsize)
        self._Zs = np.arange(-self._fov[2] / 2 + vsize / 2, self._fov[2] / 2, vsize)

    def get_location(self,inds):
        return self._Xs[inds[0]], self._Ys[inds[1]], self._Zs[inds[2]]

    def get_fov(self):
        return self._fov

    def get_vsize(self):
        return self._vsize

    def get_T1map(self):
        return self._T1map

    def get_T2map(self):
        return self._T2map

    def get_PDmap(self):
        return self._PDmap

    def get_shape(self):
        return np.shape(self._PDmap)

    def get_params(self,indx):
        return self._PDmap[indx],self._T1map[indx],self._T2map[indx]

    def get_list_locs(self):
        list_locs = []
        for x in self._Xs:
            for y in self._Ys:
                for z in self._Zs:
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
    def __init__(self,type_map,type_params,vsize,dBmap=0):
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


        super().__init__(T1map,T2map,PDmap,vsize,dBmap)



def makeSphericalPhantom(n,fov,T1s,T2s,PDs,radii):
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

    return DTTPhantom(type_map,type_params,vsize)


def makePlanarPhantom(n,fov,T1s,T2s,PDs,radii):
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

    return DTTPhantom(type_map, type_params, vsize)