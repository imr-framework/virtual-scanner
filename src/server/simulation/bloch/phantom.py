# Phantom (i.e. 3D subject model with T1,T2,PD,df maps) functionalities

import numpy as np


class Phantom:
    def __init__(self,T1map,T2map,PDmap,vsize,dBmap=0):
        self._T1map = T1map
        self._T2map = T2map
        self._PDmap = PDmap
        self._vsize = vsize
        self._dBmap = dBmap

        # Check shape consistency
        # Find field-of-view
        self._fov = vsize*np.array(np.shape(T1map))

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
        T1map = np.zeros(np.shape(type_map))
        T2map = np.zeros(np.shape(type_map))
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

    type_params[0] = (0,0,0)
    for k in range(m):
        type_params[k+1] = (PDs[k],T1s[k],T2s[k])

    return DTTPhantom(type_map,type_params,vsize)
