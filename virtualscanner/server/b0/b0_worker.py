# Incorporate and acknowledge Halbach design code

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:17:41 2018

@author: to_reilly
"""

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

mu = 1e-7


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
                  resolution=1000, bRem=1.3, simDimensions=(0.3, 0.3, 0.2)):
    # define vacuum permeability
    mu = 1e-7

    # positioning of the magnets in a circle
    angle_elements = np.linspace(0, 2 * np.pi, numMagnets, endpoint=False)

    # Use the analytical expression for the z component of a cube magnet to estimate
    # dipole momentstrength for correct scaling. Dipole approximation only valid
    # far-ish away from magnet, comparison made at 1 meter distance.

    dip_mom = magnetization(bRem, magnetSize)

    # create array to store field data
    B0 = np.zeros((int(simDimensions[0] * resolution) + 1, int(simDimensions[1] * resolution) + 1,
                   int(simDimensions[2] * resolution) + 1, 3), dtype=np.float32)

    # create halbach array
    for row in rings:
        for angle in angle_elements:
            position = (radius * np.cos(angle), radius * np.sin(angle), row)

            dip_vec = [dip_mom * np.cos(kValue * angle), dip_mom * np.sin(kValue * angle)]
            dip_vec = np.multiply(dip_vec, mu)

            # calculate contributions of magnet to total field, dipole always points in xy plane
            # so second term is zero for the z component
            B0 += singleMagnet(position, dip_vec, simDimensions, resolution)

    return B0


def b0_halbach_worker(innerRingRadii, innerNumMagnets, numRings, ringSep, DSV):
    print('Starting simulation...')
    def fieldError(shimVector):
        field = np.zeros(np.size(sharedShimMagnetsFields, 0))
        for idx1 in range(0, np.size(shimVector)):
            field += sharedShimMagnetsFields[:, idx1, shimVector[idx1]]
        return (((np.max(field) - np.min(field)) / np.mean(field)) * 1e6,)

    outerRingRadii = innerRingRadii + 21 * 1e-3
    outerNumMagnets = innerNumMagnets + 7

    resolution = 5
    magnetLength = (numRings - 1) * ringSep
    ringPositions = np.linspace(-magnetLength / 2, magnetLength / 2, numRings)

    # population
    popSim = 10000
    maxGeneration = 50

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
        print(output_text)

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
            print(output_text)

            bestError = min(fits)
            actualBestVector = tools.selBest(pop, 1)[0]

        minTracker[g - 1] = min(fits)
        output_text += "Evaluation took " + str(time.time() - startTime) + " seconds \n"
        output_text += "Minimum: %i ppm \n" % min(fits)
        print(output_text)

    output_text += "-- End of (successful) evolution -- \n"
    best_ind = tools.selBest(pop, 1)[0]
    output_text += "Best individual is %s, %s \n" % (best_ind, best_ind.fitness.values)
    print(output_text)

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
    print(output_text)

    return minTracker, coordinateAxis, maskedField, output_text
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

def b0_plot_worker(maskedField, xslice, yslice, zslice):
    # Make plot with plotly and return them in json format
    fig = make_subplots(rows=1,cols=3)
    fig.add_trace(go.Heatmap(z=maskedField[xslice,:,:]),row=1,col=1)
    fig.add_trace(go.Heatmap(z=maskedField[:,yslice,:]),row=1,col=2)
    fig.add_trace(go.Heatmap(z=maskedField[:,:,zslice]),row=1,col=3)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


















