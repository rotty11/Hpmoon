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
 * @file ag.cpp
 * @author Juan José Escobar Pérez
 * @date 09/12/2016
 * @brief Implementation of the islands-based genetic algorithm model
 * @copyright Hpmoon (c) 2015 University of Granada
 */

/********************************* Includes *******************************/

#include "ag.h"
#include "evaluation.h"
#include <algorithm> // std::max_element
#include <numeric> // std::iota
#include <omp.h> // OpenMP
#include <set> // std::set
#include <string.h> // memcpy, memset

/********************************* Defines ********************************/

#define INITIALIZE 0
#define IGNORE_VALUE 1
#define FINISH 2

/********************************* Methods ********************************/

void printSubpopulations(const Individual *const subpops, const int nSubpopulations, const Config *const conf) {

	for (int sp = 0; sp < nSubpopulations; ++sp) {
		fprintf(stdout, "Process %d-%d: Subpop_%d\n", conf -> mpiRank, omp_get_thread_num(), sp);
		for (int i = 0; i < conf -> subpopulationSize; ++i) {
			fprintf(stdout, "Process %d: Individual %d: ", conf -> mpiRank, i);
			for (int f = 0; f < conf -> nFeatures; ++f)  {
				fprintf(stdout, " %d", subpops[i].chromosome[f]);
			}
			fprintf(stdout, " * Rank: %d", subpops[i].rank);
			fprintf(stdout, " * Fit0: %f", subpops[sp * conf -> familySize + i].fitness[0]);
			fprintf(stdout, " * Fit1: %f", subpops[sp * conf -> familySize + i].fitness[1]);
			fprintf(stdout, "* Crow: %f", subpops[i].crowding);
			fprintf(stdout, "\n");
		}
	}
	
}


/**
 * @brief Allocate memory for all subpopulations (parents and children). Also, they are initialized
 * @param conf The structure with all configuration parameters
 * @return The first subpopulations
 */
Individual* createSubpopulations(const Config *const conf) {


	/********** Initialization of the subpopulations and the individuals ***********/

	// Allocate memory for parents and children
	Individual *subpops = new Individual[conf -> totalIndividuals];
	for (int i = 0; i < conf -> totalIndividuals; ++i) {
		memset(subpops[i].chromosome, 0, conf -> nFeatures * sizeof(unsigned char));
		for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
			subpops[i].fitness[obj] = 0.0f;
		}
		subpops[i].crowding = 0.0f;
		subpops[i].rank = -1;
		subpops[i].nSelFeatures = 0;
	}

	// Only the parents of each subpopulation are initialized
	for (int it = 0; it < conf -> totalIndividuals; it += conf -> familySize) {
		for (int i = it; i < it + conf -> subpopulationSize; ++i) {

			// Set value '1' 'conf -> maxFeatures' decision variables at most
			for (int mf = 0; mf < conf -> maxFeatures; ++mf) {
				int randomFeature = rand() % conf -> nFeatures;
				if (!(subpops[i].chromosome[randomFeature] & 1)) {
					subpops[i].nSelFeatures += (subpops[i].chromosome[randomFeature] = 1);
				}
			}
		}
	}

	return subpops;
}


/**
 * @brief Tournament between randomly selected individuals. The best individuals are stored in the pool
 * @param conf The structure with all configuration parameters
 * @return The pool with the selected individuals
 */
int* getPool(const Config *const conf) {

	// Create and fill the pool
	int *pool = new int[conf -> poolSize];
	std::set<int> candidates;
	for (int i = 0; i < conf -> poolSize; ++i) {

		// std::set guarantees unique elements in the insert function
		for (int j = 0; j < conf -> tourSize; ++j) {
			candidates.insert(rand() % conf -> subpopulationSize);
		}

		// At this point, the individuals already are sorted by rank and crowding distance
		// Therefore, lower index is better
		pool[i] = *(candidates.begin());
		candidates.clear();
	}

	return pool;
}


/**
 * @brief Perform binary crossover between two individuals (uniform crossover)
 * @param subpop Current subpopulation
 * @param pool Position of the selected individuals for the crossover
 * @param conf The structure with all configuration parameters
 * @return The number of generated children
 */
int crossoverUniform(Individual *const subpop, const int *const pool, const Config *const conf) {

	// Reset the children
	for (int i = conf -> subpopulationSize; i < conf -> familySize; ++i) {
		memset(subpop[i].chromosome, 0, conf -> nFeatures * sizeof(unsigned char));
		for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
			subpop[i].fitness[obj] = 0.0f;
		}
		subpop[i].nSelFeatures = 0;
		subpop[i].rank = -1;
		subpop[i].crowding = 0.0f;
	}

	Individual *child = &(subpop[conf -> subpopulationSize]);
	for (int i = 0; i < conf -> poolSize; ++i) {

		// 75% probability perform crossover. Two childen are generated
		Individual *parent1 = &(subpop[pool[rand() % conf -> poolSize]]);
		if ((rand() / (float) RAND_MAX) < 0.75f) {

			// Avoid repeated parents
			Individual *parent2 = &(subpop[pool[rand() % conf -> poolSize]]);
			Individual *child2 = child + 1;
			while (parent1 == parent2) {
				parent2 = &(subpop[pool[rand() % conf -> poolSize]]);
			}

			// Perform uniform crossover for each decision variable in the chromosome
			for (int f = 0; f < conf -> nFeatures; ++f) {

				// 50% probability perform copy the decision variable of the other parent
				if ((parent1 -> chromosome[f] != parent2 -> chromosome[f]) && ((rand() / (float) RAND_MAX) < 0.5f)) {
					child -> nSelFeatures += (child -> chromosome[f] = parent2 -> chromosome[f]);
					child2 -> nSelFeatures += (child2 -> chromosome[f] = parent1 -> chromosome[f]);
				}
				else {
					child -> nSelFeatures += (child -> chromosome[f] = parent1 -> chromosome[f]);
					child2 -> nSelFeatures += (child2 -> chromosome[f] = parent2 -> chromosome[f]);
				}
			}

			// At least one decision variable must be set to '1'
			if (child -> nSelFeatures == 0) {
				child -> chromosome[rand() % conf -> nFeatures] = child -> nSelFeatures = 1;
			}

			if (child2 -> nSelFeatures == 0) {
				child2 -> chromosome[rand() % conf -> nFeatures] = child2 -> nSelFeatures = 1;
			}
			child += 2;
		}

		// 25% probability perform mutation. One child is generated
		// Mutation is based on random mutation
		else {

			// Perform mutation on each element of the selected parent
			for (int f = 0; f < conf -> nFeatures; ++f) {

				// 10% probability perform mutation (gen level)
				float probability = (rand() / (float) RAND_MAX);
				if (probability < 0.1f) {
					if ((rand() / (float) RAND_MAX) > 0.01f) {
						child -> chromosome[f] = 0;
					}
					else {
						child -> nSelFeatures += (child -> chromosome[f] = 1);
					}
				}
				else {
					child -> nSelFeatures += (child -> chromosome[f] = parent1 -> chromosome[f]);
				}
			}

			// At least one decision variable must be set to '1'
			if (child -> nSelFeatures == 0) {
				child -> chromosome[rand() % conf -> nFeatures] = child -> nSelFeatures = 1;
			}
			++child;
		}
	}

	return child - (subpop + conf -> subpopulationSize);
}


/**
 * @brief Perform the migrations between subpopulations
 * @param subpops The subpopulations
 * @param nSubpopulations The number of subpopulations involved in the migration
 * @param nIndsFronts0 The number of individuals in the front 0 of each subpopulation
 * @param conf The structure with all configuration parameters
 */
void migration(Individual *const subpops, const int nSubpopulations, const int *const nIndsFronts0, const Config *const conf) {

	// From subpopulations randomly choosen some individuals of the front 0 are copied to each subpopulation (the worst individuals are deleted)
	for (int subpop = 0; subpop < nSubpopulations; ++subpop) {

		// This vector contains the available subpopulations indexes which are randomly choosen for copy the individuals of the front 0
		std::vector<int> randomIndex(nSubpopulations);
		std::iota(randomIndex.begin(), randomIndex.end(), 0);

		// The current subpopulation will not copy its own individuals
		randomIndex.erase(randomIndex.begin() + subpop);
		std::random_shuffle(randomIndex.begin(), randomIndex.end());

		int maxCopy = conf -> subpopulationSize - nIndsFronts0[subpop];
		Individual *ptrDest = subpops + (subpop * conf -> familySize) + conf -> subpopulationSize;
		for (int subpop2 = 0; subpop2 < nSubpopulations - 1 && maxCopy > 0; ++subpop2) {
			int toCopy = std::min(maxCopy, nIndsFronts0[randomIndex[subpop2]] >> 1);
			Individual *ptrOrig = subpops + (randomIndex[subpop2] * conf -> familySize);
			ptrDest -= toCopy;
			memcpy(ptrDest, ptrOrig, toCopy * sizeof(Individual));
			maxCopy -= toCopy;
		}
	}

	#pragma omp parallel for
	for (int sp = 0; sp < nSubpopulations; ++sp) {
		int popIndex = sp * conf -> familySize;

		// The crowding distance of the subpopulation is initialized again for the next nonDominationSort
		for (int i = popIndex;  i < popIndex + conf -> subpopulationSize; ++i) {
			subpops[i].crowding = 0.0f;
		}
		nonDominationSort(subpops + popIndex, conf -> subpopulationSize, conf);
	}
}


/**
 * @brief Evolve one subpopulation running in different modes: Sequential, CPU or GPU only and Heterogeneous (full cooperation between all available devices)
 * @param subpop The subpopulation to be evolved
 * @param nIndsFronts0 The number of individuals in the front 0 of the subpopulation
 * @param devicesObject Structure containing the information of a device
 * @param trDataBase The training database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param conf The structure with all configuration parameters
 * @param initialize If the subpopulation must be initialized or not
 */
void evolve(Individual *const subpop, int *const nIndsFronts0, CLDevice *const devicesObject, const float *const trDataBase, const int *const selInstances, const Config *const conf, const bool initialize) {


	/********** Multi-objective individuals evaluation over all subpopulations ***********/

	int nDevices = (omp_get_num_threads() > 1) ? 1 : conf -> nDevices;
	if (initialize) {
		evaluation(subpop, conf -> subpopulationSize, devicesObject, nDevices, trDataBase, selInstances, conf);


		/********** Sort the subpopulation with the 'Non-dominated sorting' method ***********/

		nIndsFronts0[0] = nonDominationSort(subpop, conf -> subpopulationSize, conf);
	}


	/********** Start the evolution process ***********/

	for (int g = 0; g < conf -> nGenerations; ++g) {


		/********** Fill the mating pool and perform crossover ***********/

		const int *const pool = getPool(conf);
		int nChildren = crossoverUniform(subpop, pool, conf);

		// Local resources used are released
		delete[] pool;


		/********** Multi-objective individuals evaluation over the subpopulation ***********/

		evaluation(subpop + conf -> subpopulationSize, nChildren, devicesObject, nDevices, trDataBase, selInstances, conf);


		/********** The crowding distance of the parents is initialized again for the next nonDominationSort ***********/

		for (int i = 0;  i < conf -> subpopulationSize; ++i) {
			subpop[i].crowding = 0.0f;
		}

		// Replace subpopulation
		// Parents and children are sorted by rank and crowding distance.
		// The first 'conf -> subpopulationSize' individuals will continue for the next generation
		nIndsFronts0[0] = nonDominationSort(subpop, conf -> subpopulationSize + nChildren, conf);
	}
}


/**
 * @brief Island-based genetic algorithm model
 * @param subpops The initial subpopulations
 * @param devicesObject Structure containing the OpenCL variables of a device
 * @param trDataBase The training database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param conf The structure with all configuration parameters
 */
void agIslands(Individual *subpops, CLDevice *const devicesObject, const float *const trDataBase, const int *const selInstances, const Config *const conf) {


	/********** MPI variables ***********/

	MPI::Status status;
	int array_of_blocklengths[3] = {conf -> nFeatures, conf -> nObjectives + 1, 2};
	MPI::Datatype array_of_types[3] = {MPI::UNSIGNED_CHAR, MPI::FLOAT, MPI::INT};

	// The 'Individual' datatype must be converted to a MPI datatype and commit it
	MPI::Aint array_of_displacement[3] = {offsetof(Individual, chromosome), offsetof(Individual, fitness), offsetof(Individual, rank)};
	MPI::Datatype Individual_MPI_type = MPI::Datatype::Create_struct(3, array_of_blocklengths, array_of_displacement, array_of_types);
	Individual_MPI_type.Commit();


	/******* Measure and start the master-worker algorithm *******/

	MPI::COMM_WORLD.Barrier();

	// Master
	if (conf -> mpiRank == 0) {
		double timeStart = omp_get_wtime();
		MPI::Request requests[conf -> mpiSize - 1];
		int nIndsFronts0[conf -> nSubpopulations];
		int finalFront0;

		// I work alone
		if (conf -> mpiSize == 1) {
			omp_set_nested(1);
			int nThreads = std::min(conf -> nDevices, conf -> nSubpopulations);
			for (int gMig = 0; gMig < conf -> nGlobalMigrations; ++gMig) {

				#pragma omp parallel for num_threads(nThreads) schedule(dynamic, 1)
				for (int sp = 0; sp < conf -> nSubpopulations; ++sp) {
					int popIndex = sp * conf -> familySize;
					evolve(subpops + popIndex, &nIndsFronts0[sp], &devicesObject[omp_get_thread_num()], trDataBase, selInstances, conf, gMig == 0);
				}

				// Migration process between subpopulations
				if (gMig != conf -> nGlobalMigrations - 1 && conf -> nSubpopulations > 1) {
					migration(subpops, conf -> nSubpopulations, nIndsFronts0, conf);
				}
			}
		}

		// I need to distribute
		else {

			// The master receives the number of subpopulations that each worker can process
			int workerCapacities[conf -> mpiSize - 1];
			for (int p = 1; p < conf -> mpiSize; ++p) {
				requests[p - 1] = MPI::COMM_WORLD.Irecv(&workerCapacities[p - 1], 1, MPI::INT, p, MPI::ANY_TAG);
			}

			MPI::Request::Waitall(conf -> mpiSize - 1, requests);


			/********** In each migration the individuals are exchanged between subpopulations of different nodes  ***********/

			for (int gMig = 0; gMig < conf -> nGlobalMigrations; ++gMig) {

				// Send some work to the workers
				int nextWork = 0;
				int sent = 0;
				int mpiTag = (gMig == 0) ? INITIALIZE : IGNORE_VALUE;
				for (int p = 1; p < conf -> mpiSize && nextWork < conf -> nSubpopulations; ++p) {
						int finallyWork = std::min(workerCapacities[p - 1], conf -> nSubpopulations - nextWork);
						int popIndex = nextWork * conf -> familySize;
						requests[p - 1] = MPI::COMM_WORLD.Isend(subpops + popIndex, finallyWork * conf -> familySize, Individual_MPI_type, p, mpiTag);
						nextWork += finallyWork;
						++sent;
				}
				MPI::Request::Waitall(sent, requests);

				// Dynamically distribute the subpopulations
				int receivedPtr = 0;
				while (nextWork < conf -> nSubpopulations) {
					MPI::COMM_WORLD.Recv(subpops + (receivedPtr * conf -> familySize), conf -> familySize, Individual_MPI_type, MPI::ANY_SOURCE, MPI::ANY_TAG, status);
					int popIndex = nextWork * conf -> familySize;
					MPI::COMM_WORLD.Send(subpops + popIndex, conf -> familySize, Individual_MPI_type, status.Get_source(), mpiTag);
					nIndsFronts0[receivedPtr] = status.Get_tag();
					++receivedPtr;
					++nextWork;
				}

				// Receive the remaining work
				while (receivedPtr < conf -> nSubpopulations) {
					MPI::COMM_WORLD.Recv(subpops + (receivedPtr * conf -> familySize), conf -> familySize, Individual_MPI_type, MPI::ANY_SOURCE, MPI::ANY_TAG, status);
					MPI::COMM_WORLD.Send(NULL, 0, MPI::INT, status.Get_source(), FINISH);
					nIndsFronts0[receivedPtr] = status.Get_tag();
					++receivedPtr;
				}

				// Migration process between subpopulations of different nodes
				if (gMig != conf -> nGlobalMigrations - 1 && conf -> nSubpopulations > 1) {
					migration(subpops, conf -> nSubpopulations, nIndsFronts0, conf);
				}
			}

			// Notify to all workers that the work has finished
			for (int p = 1; p < conf -> mpiSize; ++p) {
				requests[p - 1] = MPI::COMM_WORLD.Isend(NULL, 0, MPI::INT, p, FINISH);
			}
		}


		/********** Recombination process ***********/

		if (conf -> nSubpopulations > 1) {
			for (int sp = 0; sp < conf -> nSubpopulations; ++sp) {
				memcpy(subpops + (sp * conf -> subpopulationSize), subpops + (sp * conf -> familySize), conf -> subpopulationSize * sizeof(Individual));
			}

			// The crowding distance of the subpopulation is initialized again for the next nonDominationSort
			#pragma omp parallel for
			for (int i = 0;  i < conf -> worldSize; ++i) {
				subpops[i].crowding = 0.0f;
			}
			finalFront0 = std::min(conf -> subpopulationSize, nonDominationSort(subpops, conf -> worldSize, conf));
		}
		else {
			finalFront0 = nIndsFronts0[0];
		}

		// All processes must reach this point in order to provide a real time measure
		MPI::Request::Waitall(conf -> mpiSize - 1, requests);
		MPI::COMM_WORLD.Barrier();
		fprintf(stdout, "%.10g\n", (omp_get_wtime() - timeStart) * 1000.0);

		// Get the hypervolume
		fprintf(stdout, "%.6g\n", getHypervolume(subpops, finalFront0, conf));

		// Generation of the Gnuplot file for display the Pareto front
		generateDataPlot(subpops, finalFront0, conf);
		generateGnuplot(conf);
	}

	// Workers
	else {

		// The worker tells to the master how many subpopulations can be processed
		MPI::COMM_WORLD.Isend(&(conf -> nDevices), 1, MPI::INT, 0, 0);
		omp_set_nested(1);
		subpops = new Individual[conf -> nDevices * conf -> familySize];

		// The worker receives as many subpopulations as number of devices at most
		MPI::COMM_WORLD.Recv(subpops, conf -> nDevices * conf -> familySize, Individual_MPI_type, 0, MPI::ANY_TAG, status);

		while (status.Get_tag() != FINISH) {
			int nSubpopulations = status.Get_count(Individual_MPI_type) / conf -> familySize;
			int EXIT = false;

			#pragma omp parallel num_threads(nSubpopulations)
			{
				int threadID = omp_get_thread_num();
				MPI::Request request;
				MPI::Status stat = status;
				int nIndsFronts0;
				int popIndex = threadID * conf -> familySize;
				do {
					evolve(subpops + popIndex, &nIndsFronts0, &devicesObject[threadID], trDataBase, selInstances, conf, stat.Get_tag() == INITIALIZE);

					// The Worker sends to the master the subpopulations already evaluated and will request new work
					request = MPI::COMM_WORLD.Isend(subpops + popIndex, conf -> familySize, Individual_MPI_type, 0, nIndsFronts0);
					request.Wait();
					MPI::COMM_WORLD.Recv(subpops + popIndex, conf -> familySize, Individual_MPI_type, 0, MPI::ANY_TAG, stat);
				} while (stat.Get_tag() != FINISH);
			}

			MPI::COMM_WORLD.Recv(subpops, conf -> nDevices * conf -> familySize, Individual_MPI_type, 0, MPI::ANY_TAG, status);
		}

		// All process must reach this point in order to provide a real time measure
		MPI::COMM_WORLD.Barrier();
	}

	// Variables used by both master and workers are released
	Individual_MPI_type.Free();
}
