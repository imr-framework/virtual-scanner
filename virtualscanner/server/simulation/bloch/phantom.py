# Copyright of the Board of Trustees of Columbia University in the City of New York
"""
Numerical phantom generation and access
"""

import numpy as np
import scipy.signal as ss
import h5py
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat

# TODO : Add T2* into the Class
class Phantom:
    """Generic numerical phantom for MRI simulations

    The phantom is mainly defined by three matrices of T1, T2, and PD values, respectively.
    At the moment, each index in the matrix corresponds to a single spin group.
    The overall physical size is determined by vsize; phantom voxels must be isotropic.

    Parameters
    ----------
    T1map : numpy.ndarray
        Matrix of T1 values in seconds
    T2map : numpy.ndarray
        Matrix of T2 values in seconds
    PDmap : numpy.ndarray
        Matrix PD values between 0 and 1
    vsize : float
        Voxel size in meters (isotropic)
    dBmap : numpy.ndarray, optional
        Matrix of B0 magnetic field variation across phantom
        The default is 0 and means no variation
    loc : tuple, optional
        Overall location of phantom in meters; default is (0,0,0)

    Attributes
    ----------
    fov : numpy.ndarray
        [fov_x, fov_y, fov_z]
        1 x 3 array of fields-of-view in x, y, and z directions
    Xs : numpy.ndarray
        1D array of all x locations in phantom
    Ys : numpy.ndarray
        1D array of all y locations in phantom
    Zs : numpy.ndarray
        1D array of all z locations in phantom

    """
    def __init__(self,T1map,T2map,PDmap,vsize,T2star_map=0, dBmap=0,Dmap=0,loc=(0,0,0)):
        self.vsize = vsize
        self.T1map = T1map
        self.T2map = T2map
        self.PDmap = PDmap
        self.vsize = vsize
        self.dBmap = dBmap
        self.Dmap = Dmap
        self.loc = loc

        if len(np.shape(T2star_map))== 1:
            self.T2star_map = np.zeros(self.T2map.shape)
        else:
            self.T2star_map = T2star_map


        # Find field-of-view
        self.fov = vsize*np.array(np.shape(T1map))

        # Make location vectors
        ph_shape = np.shape(self.PDmap)

        # Define coordinates
        self.Xs = self.loc[0]+np.arange(-self.fov[0] / 2 + vsize / 2, self.fov[0] / 2, vsize)
        self.Ys = self.loc[1]+np.arange(-self.fov[1] / 2 + vsize / 2, self.fov[1] / 2, vsize)
        self.Zs = self.loc[2]+np.arange(-self.fov[2] / 2 + vsize / 2, self.fov[2] / 2, vsize)

    def get_location(self,indx):
        """Returns (x,y,z) physical location in meters at given indices

        Parameters
        ----------
        indx : tuple or array_like
            (ind1, ind2, ind3)
            Index for querying

        Returns
        -------
        x, y, z : float
            physical location corresponding to index

        """
        return self.Xs[indx[0]], self.Ys[indx[1]], self.Zs[indx[2]]

    def get_shape(self):
        """Returns the phantom's matrix size

        Returns
        -------
        shape : tuple
            The matrix size in three dimensions

        """
        return np.shape(self.PDmap)

    def get_params(self,indx):
        """Returns PD, T1, and T2 at given indices

        Parameters
        ----------
        indx : tuple
            Index for querying

        Returns
        -------
        PD, T1, T2 : float
            Tissue parameters corresponding to the queried index

        """
        return self.PDmap[indx],self.T1map[indx],self.T2map[indx]

    def get_t2star(self,indx):
        """Return T2* at given indices

        Parameters
        ----------
        indx : tuple
            Index for querying

        Returns
        -------
        float
            Tissue T2* at the queried index
        """

        return self.T2star_map[indx]


    def get_diffusion_coeff(self, indx):
        """Returns D at given indices

         Parameters
         ----------
         indx : tuple
             Index for querying

         Returns
         -------
         PD, T1, T2 : float
             Tissue parameters corresponding to the queried index

         """
        if self.Dmap.size == 1:
            return self.Dmap
        else:
            return self.Dmap[indx]

    def get_list_locs(self):
        """Returns a flattened 1D array of all location vectors [(x1,y1,z1),...,(xk,yk,zk)]

        Returns
        -------
        list_locs : list

        """
        list_locs = []
        for x in self.Xs:
            for y in self.Ys:
                for z in self.Zs:
                    list_locs.append((x, y, z))
        return list_locs

    def get_list_inds(self):
        """Returns a flattened 1D array of all indices in phantom [(u1,v1,w1),...,(uk,vk,wk)]

        Returns
        -------
        list_inds : list

        """
        list_inds = []
        sh = self.get_shape()
        for u in range(sh[0]):
            for v in range(sh[1]):
                for w in range(sh[2]):
                    list_inds.append((u,v,w))
        return list_inds

    def output_mat(self, output_folder, name='phantom'):
        savemat(f'{output_folder}/{name}.mat', {'T1map': self.T1map, 'T2map': self.T2map, 'PDmap': self.PDmap,
                                              'vsize': self.vsize})
        return


    # TODO : add T2star into the h5 -> JEMRIS incorporation
    def output_h5(self, output_folder, name='phantom'):
        """
        Inputs
        ------
        output_folder : str
            Folder in which to output the h5 file
        """

        GAMMA = 2 * 42.58e6 * np.pi

        pht_shape = list(np.flip(self.get_shape()))
        dim = len(pht_shape)

        pht_shape.append(5)


        PDmap_au = np.swapaxes(self.PDmap,0,-1)
        T1map_ms = np.swapaxes(self.T1map * 1e3, 0,-1)
        T2map_ms = np.swapaxes(self.T2map * 1e3,0,-1)

        T1map_ms_inv = np.where(T1map_ms > 0, 1/T1map_ms, 0)
        T2map_ms_inv = np.where(T2map_ms > 0, 1/T2map_ms, 0)


        if np.shape(self.dBmap) == tuple(pht_shape): # TODO what is this doing???
            dBmap_rad_per_ms = np.swapaxes(self.dBmap * GAMMA * 1e-3, 0, -1)
        elif self.dBmap == 0:
            dBmap_rad_per_ms = np.zeros(np.flip(self.get_shape()))
        else:
            dBmap_rad_per_ms = self.dBmap * GAMMA * 1e-3




        if len(output_folder) > 0:
            output_folder += '/'
        pht_file = h5py.File(output_folder + name + '.h5', 'a')
        if "sample" in pht_file.keys():
            del pht_file["sample"]

        sample = pht_file.create_group('sample')

        data = sample.create_dataset('data', tuple(pht_shape),
                                     dtype='f')  # M0, 1/T1 [1/ms], 1/T2 [1/ms], 1/T2* [1/ms], chemical shift [rad/ms]
        offset = sample.create_dataset('offset', (3, 1), dtype='f')
        resolution = sample.create_dataset('resolution', (3, 1), dtype='f')

        if dim == 1:
            data[:, 0] = PDmap_au
            #data[:, 1] = 1 / T1map_ms
            data[:, 1] = T1map_ms_inv
            #data[:, 2] = 1 / T2map_ms
            data[:, 2] = T2map_ms_inv
            #data[:, 3] = 1 / T2map_ms  # T2 assigned as T2* for now
            data[:, 3] = T2map_ms_inv
            data[:, 4] = dBmap_rad_per_ms

        elif dim == 2:
            data[:, :, 0] = PDmap_au
            #data[:, :, 1] = 1 / T1map_ms
            #data[:, :, 2] = 1 / T2map_ms
            #data[:, :, 3] = 1 / T2map_ms  # T2 assigned as T2* for now
            data[:, :, 1] = T1map_ms_inv
            data[:, :, 2] = T2map_ms_inv
            data[:, :, 3] = T2map_ms_inv
            data[:, :, 4] = dBmap_rad_per_ms

        elif dim == 3:
            data[:, :, :, 0] = PDmap_au
            #data[:, :, :, 1] = 1 / T1map_ms
            #data[:, :, :, 2] = 1 / T2map_ms
            #data[:, :, :, 3] = 1 / T2map_ms  # T2 assigned as T2* for now
            data[:, :, :, 1] = T1map_ms_inv
            data[:, :, :, 2] = T2map_ms_inv
            data[:, :, :, 3] = T2map_ms_inv
            data[:, :, :, 4] = dBmap_rad_per_ms


        offset[:,0] = np.array(self.loc)*1000 # meters to mm conversion
        resolution[:,0] = [self.vsize*1000]*3 # isotropic

        pht_file.close()

        return

class DTTPhantom(Phantom):
    """Discrete tissue type phantom

    Phantom constructed from a finite set of tissue types and their parameters

    Parameters
    ----------
    type_map : numpy.ndarray
        Matrix of integers that map to tissue types
    type_params : dict
        Dictionary that maps tissue type number to tissue type parameters (PD,T1,T2)
    vsize : float
        Voxel size in meters (isotropic)
    dBmap : numpy.ndarray, optional
        Matrix of B0 magnetic field variation across phantom
        The default is 0 and means no variation
    loc : tuple, optional
        Overall location of phantom; default is (0,0,0)

    """

    def __init__(self,type_map,type_params,vsize,dBmap=0,Dmap=0,loc=(0,0,0)):
        print(type(type_map))
        self.type_map = type_map
        self.type_params = type_params
        T1map = np.ones(np.shape(type_map))
        T2map = np.ones(np.shape(type_map))
        PDmap = np.zeros(np.shape(type_map))

        for x in range(np.shape(type_map)[0]):
            for y in range(np.shape(type_map)[1]):
                for z in range(np.shape(type_map)[2]):
                    PDmap[x,y,z] = type_params[type_map[x,y,z]][0]
                    T1map[x,y,z] = type_params[type_map[x,y,z]][1]
                    T2map[x,y,z] = type_params[type_map[x,y,z]][2]

        super().__init__(T1map,T2map,PDmap,vsize,dBmap,Dmap,loc)

    def output_mat(self, output_folder, name='phantom'):
        types = np.array(list(self.type_params.keys()))
        params = np.array(list(self.type_params.values()))
        savemat(f'{output_folder}/{name}.mat', {'type_map': self.type_map, 'types': types, 'params': params,
                                                'vsize': self.vsize})
        return

class BrainwebPhantom(Phantom):
    """This phantom is in development.

    """

    def __init__(self, filename,dsf=1,make2d=False,loc=0,dir='z',dBmap=0):
        dsf = int(np.absolute(dsf))
        bw_data = np.load(filename, allow_pickle=True).all()
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

class SpheresArrayPlanarPhantom(DTTPhantom):
    """2D phantom extracted from a cylinder containing spheres

    Regardless of dir, this will be an axial slice of a cylinder
    That is, the plane is constructed as a z-slice and then re-indexed to lie in the x or y plane
    The centers of spheres will correspond to locations before re-indexing

    Parameters
    ----------
    centers : list or array_like
        List of 3D physical locations of the spheres' centers
    radii : list or array_like
        List of radii for the spheres
    type_params : dict
        Dictionary that maps tissue type number to tissue type parameters (PD,T1,T2)
    fov : float
        Field of view (isotropic)
    n : int
        Matrix size
    dir : str, optional {'z','x','y'}
        Orientation of plane; default is z
    R : float, optional
        Cylinder's cross-section radius; default is half of fov
    loc : tuple, optional
        Overall location (x,y,z) of phantom from isocenter in meters
        Default is (0,0,0)


    """
    def __init__(self, centers, radii, type_params, fov, n, dir='z',R=0,loc=(0,0,0)):
        if R == 0:
            R = fov/2
        vsize = fov/n
        type_map = np.zeros((n,n,1))
        q = (n-1)/2
        centers_inds = [(np.array(c) / vsize + q) for c in centers]
        nc = len(centers)
        for r1 in range(n):
            for r2 in range(n):
                if vsize * np.sqrt((r1-q)**2+(r2-q)**2)<R:
                    type_map[r1,r2,0] = nc + 1
                for k in range(len(centers_inds)):
                    ci = centers_inds[k]
                    d = vsize * np.sqrt((r1 - ci[0]) ** 2 + (r2 - ci[1])**2)
                    if d <= radii[k]:
                        type_map[r1,r2,0] = k + 1
                        break
        if dir == 'x':
            type_map = np.swapaxes(type_map, 1, 2)
            type_map = np.swapaxes(type_map, 0, 1)
        elif dir == 'y':
            type_map = np.swapaxes(type_map, 0, 2)
            type_map = np.swapaxes(type_map, 0, 1)

        super().__init__(type_map, type_params, vsize, loc=loc)


def makeSphericalPhantom(n,fov,T1s,T2s,PDs,radii,loc=(0,0,0)):
    """Make a spherical phantom with concentric layers

    Parameters
    ----------
    n : int
        Matrix size of phantom (isotropic)
    fov : float
        Field of view of phantom (isotropic)
    T1s : numpy.ndarray or list
        List of T1s in seconds for the layers, going outward
    T2s : numpy.ndarray or list
        List of T2s in seconds for the layers, going outward
    PDs : numpy.ndarray or list
        List of PDs between 0 and 1 for the layers, going outward
    radii : numpy.ndarray
        List of radii that define the layers
        Note that the radii are expected to go from smallest to largest
        If not, they will be sorted first without sorting the parameters
    loc : tuple, optional
        Overall (x,y,z) location of phantom; default is (0,0,0)

    Returns
    -------
    phantom : DTTPhantom

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
    """Make a circular 2D phantom with concentric layers

    Parameters
    ----------
    n : int
        Matrix size of phantom (isotropic)
    fov : float
        Field of view of phantom (isotropic)
    T1s : numpy.ndarray or list
        List of T1s in seconds for the layers, going outward
    T2s : numpy.ndarray or list
        List of T2s in seconds for the layers, going outward
    PDs : numpy.ndarray or list
        List of PDs between 0 and 1 for the layers, going outward
    radii : numpy.ndarray
        List of radii that define the layers
        Note that the radii are expected to go from smallest to largest
        If not, they will be sorted first without sorting the parameters
    dir : str, optional {'z','x','y'}
         Orientation of the plane; default is z, axial
    loc : tuple, optional
        Overall (x,y,z) location of phantom; default is (0,0,0)

    Returns
    -------
    phantom : DTTPhantom

    """
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

    return DTTPhantom(type_map=type_map, type_params=type_params, vsize=vsize, dBmap = 0, loc=loc)


def makeCylindricalPhantom(dim=2,n=16,dir='z',loc=0,fov=0.24,type_params=None):
    """Makes a cylindrical phantom with fixed geometry and T1, T2, PD but variable resolution and overall size

    The cylinder's diameter is the same as its height; three layers of spheres represent T1, T2, and PD variation.

    Parameters
    ----------
    dim : int, optional {2,3}
         Dimension of phantom created
    n : int
        Number of spin groups in each dimension; default is 16
    dir : str, optional {'z', 'x', 'y'}
        Direction (norm) of plane in the case of 2D phantom
    loc : float, optional
        Location of plane relative to isocenter; default is 0
    fov : float, optional
        Physical length for both diameter and height of cylinder

    Returns
    -------
    phantom : DTTPhantom

    """
    R = fov/2 # m
    r = R/4 # m
    h = fov # m
    s2 = np.sqrt(2)
    s3 = np.sqrt(3)
    vsize = fov/n
    plane_loc = fov/3
    centers = [(0,R/2,plane_loc),(-R*s3/4,-R/4,plane_loc),(R*s3/4,-R/4,plane_loc), # PD spheres
               (R/(2*s2),R/(2*s2),0),(-R/(2*s2),R/(2*s2),0),(-R/(2*s2),-R/(2*s2),0),(R/(2*s2),-R/(2*s2),0), # T1 spheres
               (0,R/2,-plane_loc),(-R/2,0,-plane_loc),(0,-R/2,-plane_loc),(R/2,0,-plane_loc)] # T2 spheres
    centers_inds = [(np.array(c)/vsize + (n-1)/2) for c in centers]

    if type_params is None:
        type_params = {0:(0,1,1), # background
                       1:(1,0.5,0.1),2:(0.75,0.5,0.1),3:(0.5,0.5,0.1), # PD spheres
                    4:(0.75,1.5,0.1),5:(0.75,0.6,0.1),6:(0.75,0.25,0.1),7:(0.75,0.1,0.1), # T1 spheres
                    8:(0.75,0.5,0.5),9:(0.75,0.5,0.15),10:(0.75,0.5,0.05),11:(0.75,0.5,0.01), # T2 spheres
                    13:(0.25,0.5,0.1)}

    q = (n - 1) / 2
    p = 'xyz'.index(dir)
    pht_loc = (0, 0, 0)

    if dim == 3:
        type_map = np.zeros((n, n, n))
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    if vsize*np.sqrt((x-q)**2 + (y-q)**2) < R:
                        type_map[x,y,z] = 13
                    for k in range(len(centers_inds)):
                        ci = centers_inds[k]
                        d = vsize*np.sqrt((x-ci[0])**2+(y-ci[1])**2+(z-ci[2])**2)
                        if d <= r:
                            type_map[x, y, z] = k + 1
                            break

    elif dim == 2:
        pht_loc = np.roll((loc,0,0),p)
        # 2D phantom
        type_map = np.zeros(np.roll((1,n,n),p))
        for r1 in range(n):
            for r2 in range(n):
                x,y,z = np.roll([q+loc/vsize,r1,r2],p)
                u,v,w = np.roll((0,r1,r2),p)
                if vsize*np.sqrt((x-q)**2 + (y-q)**2) < R:
                    type_map[u,v,w] = 13
                    for k in range(len(centers_inds)):
                        ci = centers_inds[k]
                        d = vsize*np.sqrt((x-ci[0])**2+(y-ci[1])**2+(z-ci[2])**2)
                        if d <= r:
                            type_map[u,v,w] = k + 1
                            break


    else:
        raise ValueError('#Dimensions must be 2 or 3')

    phantom = DTTPhantom(type_map, type_params, vsize, loc=(0,0,0))
    return phantom

def makeCustomCylindricalPhantom(T1T2PD0, PDs, T1s, T2s, T1T2PD1=None, dim=2,n=16,dir='z',loc=0,fov=0.25):
    """ Make a cylindrical phantom with the 3 planes, same as in makeCylindricalPhantom()
             but allow custom T1/T2/PD values

    Params
    ------
    T1T2PD0 : array_like
        [T1(seconds),T2(seconds),PD(a.u.)] for all spheres whenever that parameter is not varied
    PDs : array_like
        Length-3 array of proton density values (a.u.) of the 3 spheres on the PD-plane
    T1s : array_like
        Length-4 array of T1 values (seconds) of the 4 spheres on the T1-plane
    T2s : array_like
        Length-4 array of T2 values (seconds) of the 4 spheres on the T2-plane
    T1T2PD1 : array_like
        [T1(seconds),T2(seconds),PD(a.u.)] for the main body of the cylinder surrounding the spheres
    dim : int, optional {2,3}
         Dimension of phantom created
    n : int
        Number of spin groups in each dimension; default is 16
    dir : str, optional {'z', 'x', 'y'}
        Direction (norm) of plane in the case of 2D phantom
    loc : float, optional
        Location of plane relative to isocenter; default is 0
    fov : float, optional
        Physical length for both diameter and height of cylinder

    Returns
    -------
    phantom : DTTPhantom

    """
    T1o, T2o, PDo = tuple(T1T2PD0)
    if T1T2PD1 is None:
        T1T2PD1 = (1,4,2) # Water/CSF
    T11, T21, PD1 = tuple(T1T2PD1)
    type_params_custom = {0:(0,0,0), # background has zero proton density
                       1:(PDs[0],T1o,T2o),2:(PDs[1],T1o,T2o),3:(PDs[2],T1o,T2o), # PD spheres
                    4:(PDo,T1s[0],T2o),5:(PDo,T1s[1],T2o),6:(PDo,T1s[2],T2o),7:(PDo,T1s[3],T2o), # T1 spheres
                    8:(PDo,T1o,T2s[0]),9:(PDo,T1o,T2s[1]),10:(PDo,T1o,T2s[2]),11:(PDo,T1o,T2s[3]), # T2 spheres
                    13:(PD1,T11,T21)} # main fill of cylinder - default is water

    phantom = makeCylindricalPhantom(dim=dim,n=n,dir=dir,loc=loc,fov=fov,type_params=type_params_custom)
    return phantom




if __name__ == '__main__':
    # pht = makeCylindricalPhantom(dim=2, n=16, dir='z', loc=0, fov = 0.25)
    # #plt.imshow(pht.PDmap)
    # #plt.show()
    # #print(pht.loc)
    # #print(np.array(list(pht.type_params.keys())))
    # #print(np.array(pht.type_params.items()))
    # #print(np.array(list(pht.type_params.values()))[0])
    # pht.output_mat(output_folder='sim/amri_debug',name='cylindrical_upload_test')
    # q = loadmat('sim/amri_debug/cylindrical_upload_test.mat')
    # print(np.shape(q['types']))
    # print(np.shape(q['params']))
    # type_params = dict((q['types'][0,u],tuple(q['params'][u,:])) for u in range(q['types'].shape[1]))
    # print(type_params)
    #
    #
    # p = DTTPhantom(type_map=q['type_map'], type_params=type_params, vsize=float(q['vsize']), dBmap=0,
    #                    loc=(0, 0, 0))
    #
    # T1T2PD0 = [0.5,0.5,0.05] #
    # PDs = [1,0.5,0.25] # varying PD
    # T1s = [1,0.5,0.2,0.08]
    # T2s = [0.25,0.12,0.06,0.02]
    # mypht = makeCustomCylindricalPhantom(T1T2PD0, PDs, T1s, T2s, T1T2PD1=None, dim=2, n=16, dir='z', loc=-0.08, fov=0.24)
    #

    Npht = 16
    FOV= 0.064
    # Choose your own T1s, T2s, and PDs to fill the planes
    T1T2PD0 = [0.5, 0.1, 1]  # Default values (T1o,T2o,PDo) for use when the other parameters are varied
    PDs = [1, 0.75, 0.5]  # varying PD (3 values/spheres; a.u. relative to water=1)
    T1s = 0.25 * np.array([1.5, 0.6, 0.25, 0.1])  # varying T1 (4 values/spheres in [seconds])
    T2s = [0.5, 0.15, 0.05, 0.01]  # varying T2 (4 values/spheres in [seconds])
    T1T2PD1 = [0.5, 0.1, 1]  # Values for the main cylinder fill surrounding the sphere s
    p = makeCustomCylindricalPhantom(T1T2PD0, PDs, T1s, T2s, T1T2PD1=T1T2PD1, dim=3, n=Npht, loc=0, fov=FOV)

    # Display phantom maps (3D)

    nz_disp = 3

    print(p.T2map[:,:,3])

    plt.figure(1)
    plt.subplot(131)
    plt.imshow(np.squeeze(p.T1map[:, :, nz_disp]))
    plt.colorbar()
    plt.title('T1 map (s)')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

    plt.subplot(132)
    plt.imshow(np.squeeze(p.T2map[:, :, nz_disp]))
    plt.title('T2 map (s)')
    plt.colorbar()

    frame2 = plt.gca()
    frame2.axes.get_xaxis().set_visible(False)
    frame2.axes.get_yaxis().set_visible(False)

    plt.subplot(133)
    plt.imshow(np.squeeze(p.PDmap[:, :, nz_disp]))
    plt.title('PD map (a.u.)')
    plt.colorbar()
    frame3 = plt.gca()
    frame3.axes.get_xaxis().set_visible(False)
    frame3.axes.get_yaxis().set_visible(False)

    plt.show()

