
######################################
# Halbach simulation and optimization code authorship information
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:17:41 2018
@author: to_reilly
"""
######################################



import numpy as np
import random
from deap import algorithms, base, tools, creator
import multiprocessing
import ctypes
import time
import matplotlib.pyplot as plt
import json
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.io import savemat
import time
import numpy as np
from scipy.interpolate import interpn

from skimage import io
mu = 1e-7
BREM_TEMP_COEFF = -0.13e-2
BASE_TEMP = 25

def magnetization(bRem, dimensions, shape='cube'):
    if shape == 'cube':
        dip_mom = bRem * dimensions ** 3 / (4 * np.pi * mu)
    return dip_mom


def singleMagnet(position, dipoleMoment, simDimensions, resolution):
    # create mesh coordinates
    x = np.linspace(-simDimensions[0] / 2 + position[0], simDimensions[0] / 2 + position[0],
                    int(simDimensions[0] * resolution + 1), dtype=np.float32)
    y = np.linspace(-simDimensions[1] / 2 + position[1], simDimensions[1] / 2 + position[1],
                    int(simDimensions[1] * resolution + 1), dtype=np.float32)
    z = np.linspace(-simDimensions[2] / 2 + position[2], simDimensions[2] / 2 + position[2],
                    int(simDimensions[2] * resolution + 1), dtype=np.float32)
    x, y, z = np.meshgrid(x, y, z)

    vec_dot_dip = 3 * (x * dipoleMoment[0] + y * dipoleMoment[1])

    # calculate the distance of each mesh point to magnet, optimised for speed
    # for improved memory performance move in to b0 calculations
    vec_mag = np.square(x) + np.square(y) + np.square(z)
    vec_mag_3 = np.power(vec_mag, 1.5)
    vec_mag_5 = np.power(vec_mag, 2.5)
    del vec_mag

    B0 = np.zeros((int(simDimensions[0] * resolution) + 1, int(simDimensions[1] * resolution) + 1,
                   int(simDimensions[2] * resolution) + 1, 3), dtype=np.float32)

    # calculate contributions of magnet to total field, dipole always points in xy plane
    # so second term is zero for the z component
    B0[:, :, :, 0] += np.divide(np.multiply(x, vec_dot_dip), vec_mag_5) - np.divide(dipoleMoment[0], vec_mag_3)
    B0[:, :, :, 1] += np.divide(np.multiply(y, vec_dot_dip), vec_mag_5) - np.divide(dipoleMoment[1], vec_mag_3)
    B0[:, :, :, 2] += np.divide(np.multiply(z, vec_dot_dip), vec_mag_5)

    return B0


def createHalbach(numMagnets=24, rings=(-0.075, -0.025, 0.025, 0.075), radius=0.145, magnetSize=0.0254, kValue=2,
                  resolution=1000, bRem=1.3, simDimensions=(0.3, 0.3, 0.2),temp_array=None):
    #TODO figure out how to incorporate temperature!

    # define vacuum permeability
    mu = 1e-7

    # positioning of the magnets in a circle
    angle_elements = np.linspace(0, 2 * np.pi, numMagnets, endpoint=False)

    # Use the analytical expression for the z component of a cube magnet to estimate
    # dipole momentstrength for correct scaling. Dipole approximation only valid
    # far-ish away from magnet, comparison made at 1 meter distance.

    dip_mom = magnetization(bRem, magnetSize)

    # create array to store field data
    print(f'Sim dim: {simDimensions}')
    print(f'Resolution: {resolution}')
    B0 = np.zeros((int(simDimensions[0] * resolution) + 1, int(simDimensions[1] * resolution) + 1,
                   int(simDimensions[2] * resolution) + 1, 3), dtype=np.float32)

    # create halbach array
    for count, row in enumerate(rings):
        for elind, angle in enumerate(angle_elements):
            position = (radius * np.cos(angle), radius * np.sin(angle), row)

            dip_vec = [dip_mom * np.cos(kValue * angle), dip_mom * np.sin(kValue * angle)]
            dip_vec = np.multiply(dip_vec, mu)

            # calculate contributions of magnet to total field, dipole always points in xy plane
            # so second term is zero for the z component

            # TODO how to calculate temperature factor: as a relative percentage that scales B0 - what is the base temperature?
            if temp_array is not None:
                temperature_factor = (temp_array[count,elind]-BASE_TEMP) * BREM_TEMP_COEFF
            else:
                temperature_factor = 1
            B0 += temperature_factor * singleMagnet(position, dip_vec, simDimensions, resolution)

    return B0


def b0_halbach_worker(innerRingRadii, innerNumMagnets, numRings, ringSep, DSV, max_num_gen, resolution):
    print('Starting simulation...')

    # Calculate error term
    def fieldError(shimVector):
        field = np.zeros(np.size(sharedShimMagnetsFields, 0))
        for idx1 in range(0, np.size(shimVector)):
            field += sharedShimMagnetsFields[:, idx1, shimVector[idx1]]
        return (((np.max(field) - np.min(field)) / np.mean(field)) * 1e6,)

    outerRingRadii = innerRingRadii + 21 * 1e-3  # Fixed radius difference between inner and outer at 21 mm
    outerNumMagnets = innerNumMagnets + 7 # Fixed number of magnets difference between inner and outer at 7

   # resolution = 5
    magnetLength = (numRings - 1) * ringSep
    ringPositions = np.linspace(-magnetLength / 2, magnetLength / 2, numRings)

    # population
    popSim = 10000
    maxGeneration = max_num_gen

    output_text = ""
    ###################################################################################################################
    ##########################################                               ##########################################
    ##########################################     Create spherical mask     ##########################################
    ##########################################                               ##########################################
    ###################################################################################################################

    simDimensions = (DSV, DSV, DSV)

    coordinateAxis = np.linspace(-simDimensions[0] / 2, simDimensions[0] / 2,
                                 int(1e3 * simDimensions[0] / resolution + 1))
    coords = np.meshgrid(coordinateAxis, coordinateAxis, coordinateAxis)

    mask = np.zeros(np.shape(coords[0]))
    mask[np.square(coords[0]) + np.square(coords[1]) + np.square(coords[2]) <= (DSV / 2) ** 2] = 1

    octantMask = np.copy(mask)
    octantMask[coords[0] < 0] = 0
    octantMask[coords[1] < 0] = 0
    octantMask[coords[2] < 0] = 0

    ringPositionsSymmetery = ringPositions[ringPositions >= 0]

    shimFields = np.zeros((int(np.sum(octantMask)), np.size(ringPositionsSymmetery), np.size(innerRingRadii)))

    for positionIdx, position in enumerate(ringPositionsSymmetery):
        for sizeIdx, ringSize in enumerate(innerRingRadii):
            if position == 0:
                rings = (0,)
            else:
                rings = (-position, position)
            fieldData = createHalbach(numMagnets=innerNumMagnets[sizeIdx], rings=rings,
                                                    radius=innerRingRadii[sizeIdx], magnetSize=0.012,
                                                    resolution=1e3 / resolution, simDimensions=simDimensions)
            fieldData += createHalbach(numMagnets=outerNumMagnets[sizeIdx], rings=rings,
                                                     radius=outerRingRadii[sizeIdx], magnetSize=0.012,
                                                     resolution=1e3 / resolution, simDimensions=simDimensions)
            shimFields[:, positionIdx, sizeIdx] = fieldData[octantMask == 1, 0]

    sharedShimMagnetsFields_base = multiprocessing.Array(ctypes.c_double, np.size(shimFields))
    sharedShimMagnetsFields = np.ctypeslib.as_array(sharedShimMagnetsFields_base.get_obj())
    sharedShimMagnetsFields = sharedShimMagnetsFields.reshape(np.size(shimFields, 0), np.size(shimFields, 1),
                                                              np.size(shimFields, 2))
    sharedShimMagnetsFields[...] = shimFields[...]

    random.seed()

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, np.size(sharedShimMagnetsFields, 2) - 1)

    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool,
                     np.size(sharedShimMagnetsFields, 1))

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fieldError)

    toolbox.register("mate", tools.cxTwoPoint)

    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=popSim)
    CXPB, MUTPB = 0.55, 0.4
    output_text += "Start of evolution \n"
    print(output_text)
    startTime = time.time()
    fitnesses = list(map(toolbox.evaluate, pop))
    bestError = np.inf

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    minTracker = np.zeros((maxGeneration))
    startEvolution = time.time()
    # Begin the evolution
    while g < maxGeneration:
        startTime = time.time()
        # A new generation
        g = g + 1
        output_text += "-- Generation %i -- \n" % g
        print(f"-- Generation {g} -- \n")
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        if min(fits) < bestError:
            # best in a generation is not per se best ever due to mutations, this tracks best ever
            output_text += "BEST VECTOR: \n" + str(tools.selBest(pop, 1)[0])

            bestError = min(fits)
            actualBestVector = tools.selBest(pop, 1)[0]

        minTracker[g - 1] = min(fits)
        output_text += "Evaluation took " + str(time.time() - startTime) + " seconds \n"
        output_text += "Minimum: %i ppm \n" % min(fits)

    output_text += "-- End of (successful) evolution -- \n"
    best_ind = tools.selBest(pop, 1)[0]
    output_text += "Best individual is %s, %s \n" % (best_ind, best_ind.fitness.values)

    bestVector = np.array(actualBestVector)

    shimmedField = np.zeros(np.shape(mask))
    for positionIdx, position in enumerate(ringPositionsSymmetery):
        if position == 0:
            rings = (0,)
        else:
            rings = (-position, position)
        shimmedField += createHalbach(numMagnets=innerNumMagnets[bestVector[positionIdx]], rings=rings,
                                                    radius=innerRingRadii[bestVector[positionIdx]], magnetSize=0.012,
                                                    resolution=1e3 / resolution, simDimensions=simDimensions)[..., 0]
        shimmedField += createHalbach(numMagnets=outerNumMagnets[bestVector[positionIdx]], rings=rings,
                                                    radius=outerRingRadii[bestVector[positionIdx]], magnetSize=0.012,
                                                    resolution=1e3 / resolution, simDimensions=simDimensions)[..., 0]

    mask[mask == 0] = np.nan

    maskedField = np.abs(np.multiply(shimmedField, mask))

    output_text += "Shimmed mean: %.2f mT \n" % (1e3 * np.nanmean(maskedField))
    output_text += "Shimmed homogeneity: %.4f mT \n" % (1e3 * (np.nanmax(maskedField) - np.nanmin(maskedField)))
    output_text += "Shimmed homogeneity: %i ppm \n" % (
                1e6 * ((np.nanmax(maskedField) - np.nanmin(maskedField)) / np.nanmean(maskedField)))

    return minTracker, coordinateAxis, maskedField, output_text, bestVector, ringPositionsSymmetery, simDimensions
    # plt.figure()
    # plt.plot(coordinateAxis * 1e3,
    #          maskedField[int(np.floor(np.size(maskedField, 0) / 2)), int(np.floor(np.size(maskedField, 1) / 2)), :])
    # plt.xlabel('X axis (mm)')
    # plt.ylabel('Field strength (Tesla)')
    # plt.legend()
    #
    # plt.figure()
    # plt.semilogy(minTracker)
    # plt.title("Min error Vs generations")
    # plt.xlabel("Generation")
    # plt.ylabel("Error")
    #
    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(maskedField[:, :, int(np.floor(np.size(maskedField, 2) / 2))])
    # ax[1].imshow(maskedField[:, int(np.floor(np.size(maskedField, 1) / 2)), :])
    # ax[2].imshow(maskedField[int(np.floor(np.size(maskedField, 0) / 2)), :, :])
    #
    # plt.show()

    #savemat('halbach.mat', {'maskedField': maskedField, 'coordinateAxis': coordinateAxis, 'minTracker': minTracker})

def b0_plot_worker(maskedField, coordinates, xslice, yslice, zslice):
    # Make plot with plotly and return them in json format
    print(f'length of coordinates: {len(coordinates)}')
    if len(coordinates) == 1: # isotropic
        x_coord = coordinates[0]
        y_coord = coordinates[0]
        z_coord = coordinates[0]
    else:
        x_coord = np.arange(maskedField.shape[0])
        y_coord = np.arange(maskedField.shape[1])
        z_coord = np.arange(maskedField.shape[2])

    fig = make_subplots(rows=1,cols=3)
    fig.add_trace(go.Heatmap(z=maskedField[xslice,:,:],coloraxis="coloraxis",x=y_coord*1e3,y=z_coord*1e3),row=1,col=1)
    fig.add_trace(go.Heatmap(z=maskedField[:,yslice,:],coloraxis="coloraxis",x=x_coord*1e3,y=z_coord*1e3),row=1,col=2)
    fig.add_trace(go.Heatmap(z=maskedField[:,:,zslice],coloraxis="coloraxis",x=x_coord*1e3,y=y_coord*1e3),row=1,col=3)



    fig.update_xaxes(title_text='Y (mm)',showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text='Z (mm)',showgrid=False, row=1, col=1)

    fig.update_xaxes(title_text='X (mm)', showgrid=False, row=1, col=2)
    fig.update_yaxes(title_text='Z (mm)', showgrid=False, row=1, col=2)

    fig.update_xaxes(title_text='X (mm)', showgrid=False, row=1, col=3)
    fig.update_yaxes(title_text='Y (mm)', showgrid=False, row=1, col=3)

    fig.update_layout(
        margin=dict(
            l=5,
            r=5,
            b=5,
            t=5,
            pad=0
        ))
    fig.update_coloraxes(colorbar_nticks=10)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def b0_3d_worker():
    # Inputs: design, field
    # Output: 3D graphJSON plotting it, with sliders and everything
    fig = go.Figure()
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def b0_eval_field_any(diameter, info, temperature):
    # TODO enable temperature map
    print(f"Eval b0 diam: {diameter}")
    print(f'Eval b0 info: {info}')



    # evalDimensions are (DVx, DVy, DVz) - FOV to be evaluated
    inner_ring_radii = np.squeeze(info['inner_ring_radii'])
    if 'outer_ring_radii' in info.keys():
        outer_ring_radii = np.squeeze(info['outer_ring_radii'])
    else:
        outer_ring_radii = inner_ring_radii + 21 * 1e-3

    inner_num_magnets = np.squeeze(info['inner_num_magnets'])

    if 'outer_num_magnets' in info.keys():
        outer_num_magnets = np.squeeze(info['outer_num_magnets'])
    else:
        outer_num_magnets = inner_num_magnets + 7 # Fixed number of magnets difference between inner and outer at 7

    # Need to be able to see outside the optimized DSV for a given optimization
    # Use Halbach functions to do it!
    for positionIdx, position in enumerate(np.squeeze(info['ring_position_symmetry'])):
        if position == 0:
            rings = (0,)
        else:
            rings = (-position, position)


        use_ind = np.squeeze(info['best_vector'])[positionIdx]
        res = np.squeeze(info['res_display']) # mm
        eval_dimensions = [1e-3 * diameter]*3 # m

        # TODO Parse temperature based on magnet locations
        # only single temperature value for now, but follow the structure!
        inner_temp_array = parse_b0_temperature(temperature, rings,
                                                radius=inner_ring_radii[use_ind],
                                                num_magnets=inner_num_magnets[use_ind])
        outer_temp_array = parse_b0_temperature(temperature, rings,
                                                radius=outer_ring_radii[use_ind],
                                                num_magnets=outer_num_magnets[use_ind])

        # INNER
        if positionIdx == 0: # Only the first time
            shimmedField = createHalbach(numMagnets=inner_num_magnets[use_ind], rings=rings,
                                                        radius=inner_ring_radii[use_ind], magnetSize=0.012,
                                                        resolution=1e3 / res, simDimensions=eval_dimensions, temp_array=inner_temp_array)[..., 0]
        else:
            shimmedField += createHalbach(numMagnets=inner_num_magnets[use_ind], rings=rings,
                                                        radius=inner_ring_radii[use_ind], magnetSize=0.012,
                                                        resolution=1e3 / res, simDimensions=eval_dimensions, temp_array=inner_temp_array)[..., 0]

        # OUTER
        shimmedField += createHalbach(numMagnets=outer_num_magnets[use_ind], rings=rings,
                                                    radius=outer_ring_radii[use_ind], magnetSize=0.012,
                                                    resolution=1e3 / res, simDimensions=eval_dimensions,temp_array=outer_temp_array)[..., 0]

    # Make spherical mask


    coordinates = np.linspace(-eval_dimensions[0] / 2, eval_dimensions[0] / 2,
                                 int(1e3 * eval_dimensions[0] / res + 1))
    print('coordinates:')
    print(coordinates[0],coordinates[-1])
    coords = np.meshgrid(coordinates, coordinates, coordinates)  # Same for x, y, and z

    # Spherical mask
    mask = np.zeros(np.shape(coords[0]))
    mask[np.square(coords[0]) + np.square(coords[1]) + np.square(coords[2]) <= (1e-3 * np.squeeze(info['dsv_display']) / 2) ** 2] = 1
    mask[mask == 0] = np.nan

    masked_field = np.abs(np.multiply(shimmedField,mask))

    return masked_field, coordinates



def b0_3dplot_worker(maskedField, coordinates, axis='z'):
    # Make plot with plotly and return them in json format
    # Import data
    if len(coordinates) == 1:  # isotropic
        x_coord = coordinates[0]
        y_coord = coordinates[0]
        z_coord = coordinates[0]
    else:
        x_coord = np.arange(maskedField.shape[0])
        y_coord = np.arange(maskedField.shape[1])
        z_coord = np.arange(maskedField.shape[2])


    vol = maskedField
    nx, ny, nz = vol.shape

    CMIN = np.amin(vol[~np.isnan(vol)])
    CMAX = np.amax(vol[~np.isnan(vol)])

    # Define frames
    import plotly.graph_objects as go

    if axis == 'x':
        nb_frames = nx
        yy, zz = np.meshgrid(y_coord, z_coord)
        frames = [go.Frame(data=go.Surface(x=1e3*x_coord[k]*np.ones((ny,nz)), y=1e3*yy,
                                           z=1e3*zz,
                                           surfacecolor=(np.squeeze(vol[k, :, :])),
                                           cmin=CMIN, cmax=CMAX),
                           name=str(k)  # you need to name the frame for the animation to behave properly
                           ) for k in range(nb_frames)]
        surface = go.Surface(
                    x = x_coord[0]*np.ones((ny,nz)), y=yy, z=zz,
                    surfacecolor=(np.squeeze(vol[nb_frames-1,:,:])),
                    colorscale='plasma',
                    cmin=CMIN, cmax=CMAX,
                    colorbar=dict(thickness=20, ticklen=4))
    elif axis == 'y':
        nb_frames = ny
        xx, zz = np.meshgrid(x_coord,z_coord)
        frames = [go.Frame(data=go.Surface(x=1e3*xx, y= 1e3*y_coord[k]*np.ones((nx,nz)),
                                           z=1e3*zz,
                                           surfacecolor=(np.squeeze(vol[:, k, :])),
                                           cmin=CMIN, cmax=CMAX),
                           name=str(k)  # you need to name the frame for the animation to behave properly
                           ) for k in range(nb_frames)]
        surface = go.Surface(
                    x = 1e3*xx, y= 1e3*y_coord[0]*np.ones((nx,nz)), z=1e3*zz,
                    surfacecolor=(np.squeeze(vol[:,nb_frames-1,:])),
                    colorscale='plasma',
                    cmin=CMIN, cmax=CMAX,
                    colorbar=dict(thickness=20, ticklen=4))
    elif axis == 'z':
        nb_frames = nz

        frames = [go.Frame(data=go.Surface(x=1e3*x_coord, y=1e3*y_coord,
                                           z=1e3*z_coord[k] * np.ones((nx,ny)),
                                           surfacecolor=(np.squeeze(vol[:, :, k])),
                                           cmin=CMIN, cmax=CMAX),
                           name=str(k)  # you need to name the frame for the animation to behave properly
                          )for k in range(nb_frames)]
        surface = go.Surface(
            x=x_coord, y=y_coord, z=z_coord[0]*np.ones((nx,ny)),
            surfacecolor=(np.squeeze(vol[:,:,nb_frames - 1])),
            colorscale='plasma',
            cmin=CMIN, cmax=CMAX,
            colorbar=dict(thickness=20, ticklen=4))

    else:
        raise ValueError('Axis must be x, y, or z for 3D display of B0 map!')





    # Add all frames?
    fig = go.Figure(frames=frames)

    # Add data to be displayed before animation starts
    fig.add_trace(surface)




    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title='B0 map slices (Tesla)',
        width=600,
        height=600,
        scene=dict(
            xaxis=dict(range=[1e3*x_coord[0],1e3*x_coord[-1]],autorange=False,title='x (mm)'),
            yaxis=dict(range=[1e3*y_coord[0],1e3*y_coord[-1]],autorange=False,title='y (mm)'),
            zaxis=dict(range=[1e3*z_coord[0],1e3*z_coord[-1]], autorange=False,title='z (mm)'),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON



def b0_rings_worker(innerRadii, outerRadii, ringPositionSymmetry, dsv, bestVector):
    # Generate plot of ring locations and diameters (current optimization)
    fig = go.Figure()

    bestVector = np.squeeze(bestVector)
    innerRadii = np.squeeze(innerRadii)
    outerRadii = np.squeeze(outerRadii)
    ringPositionSymmetry = np.squeeze(ringPositionSymmetry)
    axis = 'z'

    for u in range(len(ringPositionSymmetry)):
        Ri = innerRadii[bestVector[u]]
        #Ro = Ri + 21*1e-3
        Ro = outerRadii[bestVector[u]]
        # Add ring to figure at appropriate location
        Xi, Yi, Zi = make_ring_in_3d(N=100, radius=Ri, position=ringPositionSymmetry[u],orientation=axis) # Add 3D ring visualization
        fig.add_trace(go.Scatter3d(x=1e3*Xi,y=1e3*Yi,z=1e3*Zi,mode='lines',line=dict(color='darkblue',width=4)))
        Xo, Yo, Zo = make_ring_in_3d(N=100, radius=Ro, position=ringPositionSymmetry[u],
                                     orientation=axis)  # Add 3D ring visualization
        fig.add_trace(go.Scatter3d(x=1e3*Xo, y=1e3*Yo, z=1e3*Zo, mode='lines',  line=dict(color='cornflowerblue',width=4)))

        if ringPositionSymmetry[u] != 0:
            Xi, Yi, Zi = make_ring_in_3d(N=100, radius=Ri, position=-ringPositionSymmetry[u],
                                         orientation=axis)  # Add 3D ring visualization
            fig.add_trace(go.Scatter3d(x=1e3*Xi, y=1e3*Yi, z=1e3*Zi, mode='lines', line=dict(color='darkblue', width=4)))
            Xo, Yo, Zo = make_ring_in_3d(N=100, radius=Ro, position=-ringPositionSymmetry[u],
                                         orientation=axis)  # Add 3D ring visualization
            fig.add_trace(go.Scatter3d(x=1e3*Xo, y=1e3*Yo, z=1e3*Zo, mode='lines', line=dict(color='cornflowerblue', width=4)))

    fig.update_layout(showlegend=False, scene=dict(
        xaxis=dict(autorange=True, title='x (mm)'),
        yaxis=dict(autorange=True, title='y (mm)'),
        zaxis=dict(autorange=True, title='z (mm)'),
    ))


    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


def make_ring_in_3d(N, radius, position, orientation):
    # Single plotly trace of ring
    center = np.roll([position,0,0],'xyz'.index(orientation))
    dir1 = np.roll([0,1,0],'xyz'.index(orientation))
    dir2 = np.roll([0,0,1],'xyz'.index(orientation))
    phis = np.linspace(0,2*np.pi,N,endpoint=True)

    locations = np.array([center + radius * (np.cos(phi)*dir1 + np.sin(phi)*dir2) for phi in phis])
    X = locations[:,0]
    Y = locations[:,1]
    Z = locations[:,2]

    return X, Y, Z

# TODO enable spatial differences.
def parse_b0_temperature(temperature, rings, radius, num_magnets):
    # temperature [deg celsius]
    # rings : either (0) or (-pos,+pos) - in meters
    # radius : [meters]
    # num_magnets

    # Case 1 : global temperature
    if np.size(temperature) == 1: # Only 1 value
        if np.size(rings) == 1:
            magnet_temp_array = temperature * np.ones((1,num_magnets))
        else:
            magnet_temp_array = temperature * np.ones((2,num_magnets))
    elif np.size(np.shape(temperature)) == 2:  # 2D array of (ring, magnet) values
        raise NotImplementedError("Non-single value temperature maps not implemented yet.")
    else: # 3D array in space - needs to be interpolated to magnet's center locations
        raise NotImplementedError("Non-single value temperature maps not implemented yet.")
    return magnet_temp_array


def spatially_interpolate(temp_map, coordinates, locs):
    # Perform spatial interpolation of temperature map to get temperature at a list of given locations
    # temp_map : Nx x Ny x Nz sized matrix of input temperature map
    # coordinates: list of length 3: x, y, and z arrays
    # locs : magnet locations to interpolate the temperature map to; (m x 3)

    temps = interpn(points=tuple(coordinates), values=temp_map, xi=locs,bounds_error=False, fill_value=BASE_TEMP)

    return temps