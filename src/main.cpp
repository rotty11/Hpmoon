/**
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE', which is part of Hpmoon repository.
 *
 * This work has been funded by:
 *
 * Spanish 'Ministerio de Economía y Competitividad' under grants number TIN2012-32039 and TIN2015-67020-P.\n
 * Spanish 'Ministerio de Ciencia, Innovación y Universidades' under grant number PGC2018-098813-B-C31.\n
 * European Regional Development Fund (ERDF).
 *
 * @file main.cpp
 * @author Juan José Escobar Pérez
 * @date 08/07/2015
 * @brief A parallel and distributed multi-objective genetic algorithm to EEG classification
 * @copyright Hpmoon (c) 2015 EFFICOMP
 */

/********************************* Includes ********************************/

#include "bd.h"
#include "ag.h"
#include "evaluation.h"
#include "nvml.h"


/**
 * @brief Main program
 * @param argc The number of arguments of the program
 * @param argv Arguments of the program
 */
int main(const int argc, const char **argv) {


	/********** Initialize the MPI environment ***********/

	MPI::Init_thread(MPI_THREAD_MULTIPLE);


	/********** Get the configuration data from the XML file or from the command-line ***********/

	Config conf(argc, argv);
	srand((uint) time(NULL) + conf.mpiRank); // "+ rank" is necessary in MPI

	// Initialize the subpopulations and the individuals
	// Subpopulations will have the parents and children (left half and right half respectively)
	Individual *subpops = createSubpopulations(&conf);

	// Get the initial "conf.K" centroids
	int *selInstances = getCentroids(&conf);

	// Get the databases and its normalization if it is required
	const float *const trDataBase = getDataBase(&conf);
	const float *const transposedTrDataBase = transposeDataBase(trDataBase, &conf); // Transposed database


	/********** Genetic algorithm ***********/

	// Sequential, only 1 device (CPU or GPU) or heterogeneous mode if more than 1 device is available
	CLDevice *devices = createDevices(trDataBase, selInstances, transposedTrDataBase, &conf);
	agIslands(subpops, devices, trDataBase, selInstances, &conf);

	// Variables used are released
	delete[] devices;
	delete[] subpops;
	delete[] selInstances;
	delete[] trDataBase;
	delete[] transposedTrDataBase;

	// Finish the MPI environment
	MPI::Finalize();
}
