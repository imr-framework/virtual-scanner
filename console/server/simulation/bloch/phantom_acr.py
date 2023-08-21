# Make 2D, multislice numerical phantom with ACR-like slices at any resolution
# Mainly to show geometrical accuracy of sequences
import numpy as np
import matplotlib.pyplot as plt
from virtualscanner.server.simulation.bloch.phantom import DTTPhantom


# Make ACR-like phantom with more detailed features
def make_phantom_acr(N, FOV, slice_loc, shrink_factor = 1, slice_type='grid'):
    # Define tissue types
    # Define geometrical maps
    #
    diam_circ_mask = np.zeros((N,N))
    for u in range(N):
        for v in range(N):
            diam_circ_mask[u,v] = np.linalg.norm([u-N/2,v-N/2]) <= 0.5*N*shrink_factor
    if slice_type == 'grid':
        grid_mask = make_grid(N, circ_scale=shrink_factor, line_thk=N/32, num_lines=4)
    type_map = np.zeros((N,N))
    type_map = type_map + diam_circ_mask.astype(int) + (diam_circ_mask * grid_mask).astype(int)
    PDs = [0, 1, 0.5]
    T1s = [1, 4, 0.5]
    T2s = [1, 2, 0.1]
    type_params = {0:(PDs[0],T1s[0],T2s[0]), 1:(PDs[1],T1s[1],T2s[1]), 2: (PDs[2],T1s[2],T2s[2])} # 0 - background; 1 - main filling; 2 - material within grid
    vsize = FOV/N
    type_map = np.reshape(type_map, (N,N,1))
    pht = DTTPhantom(type_map,type_params,vsize,dBmap=0,loc=(0,0,0))
    return pht



def make_phantom_circle(N, FOV, slice_loc, shrink_factor = 1):
    # Define tissue types
    # Define geometrical maps
    #
    diam_circ_mask = np.zeros((N,N))
    for u in range(N):
        for v in range(N):
            diam_circ_mask[u,v] = np.linalg.norm([u-N/2,v-N/2]) <= 0.5*N*shrink_factor
    type_map = np.zeros((N,N))

    type_map = type_map + diam_circ_mask.astype(int)

    PDs = [0, 1]
    T1s = [1, 0.5]
    T2s = [1, 0.250]
    type_params = {0:(PDs[0],T1s[0],T2s[0]), 1:(PDs[1],T1s[1],T2s[1])} # 0 - background; 1 - main filling; 2 - material within grid
    vsize = FOV/N
    type_map = np.reshape(type_map, (N,N,1))
    pht = DTTPhantom(type_map,type_params,vsize,dBmap=0,loc=(0,0,0))
    return pht



def make_grid(N, circ_scale, line_thk, num_lines):
    # Make grid
    grid_mask = np.zeros((N,N))
    line_locs = np.linspace(0,N,num_lines+2,endpoint=True)
    for u in range(N):
        for v in range(N):
            for loc in line_locs:
                if abs(u-loc)<=line_thk/2 or abs(v-loc) <= line_thk/2:
                    grid_mask[u,v] = 1
    # Make circular mask
    circular_mask = np.zeros((N,N))
    for u in range(N):
        for v in range(N):
            circular_mask[u,v] = np.linalg.norm([u-N/2,v-N/2]) <= 0.5*N*circ_scale
    # Intersect
    grid_mask = grid_mask * circular_mask
    return grid_mask


if __name__ == "__main__":
    #pht = make_phantom_circle(N=32, FOV=0.25, slice_loc=0, shrink_factor=0.8)
    pht = make_phantom_acr(N=32, FOV=0.25, slice_loc=0, shrink_factor=0.8, slice_type='grid')
    #pht.output_h5(output_folder='sim/seq_validation/Revision1') # TODO (and simulate!)
    plt.figure(1)
    plt.imshow(pht.PDmap)
    plt.show()
    #grid = make_grid(N=128, circ_scale=0.8, line_thk=2, num_lines=8)
    #plt.imshow(grid)
    #plt.show()