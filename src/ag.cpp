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
 * @copyright Hpmoon (c) 2015 EFFICOMP
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

#define MIGRATION 0
#define CLOSE_COMMUNICATIONS 1

/********************************* Methods ********************************/

void printSubpopulations(const Individual *const subpops, const int nSubpopulations, const Config *const conf) {

	for (int sp = 0; sp < nSubpopulations; ++sp) {
		fprintf(stdout, "Process %d-%d: Subpop_%d\n", conf -> mpiRank, omp_get_thread_num(), sp);
		for (int i = 0; i < conf -> subpopulationSize; ++i) {
			fprintf(stdout, "Process %d: Individual %d: ", conf -> mpiRank, i);
			for (int f = 0; f < conf -> nFeatures; ++f)  {
				fprintf(stdout, " %d", subpops[sp * conf -> familySize + i].chromosome[f]);
			}
			fprintf(stdout, " * Rank: %d", subpops[sp * conf -> familySize + i].rank);
			fprintf(stdout, " * Fit0: %f", subpops[sp * conf -> familySize + i].fitness[0]);
			fprintf(stdout, " * Fit1: %f", subpops[sp * conf -> familySize + i].fitness[1]);
			fprintf(stdout, " * Crow: %f", subpops[sp * conf -> familySize + i].crowding);
			fprintf(stdout, " * nSeF: %d", subpops[sp * conf -> familySize + i].nSelFeatures);
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
	Individual *subpops = new Individual[conf -> archipelagoFamilySize];
	for (int i = 0; i < conf -> archipelagoFamilySize; ++i) {
		memset(subpops[i].chromosome, 0, conf -> nFeatures * sizeof(unsigned char));
		for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
			subpops[i].fitness[obj] = 0.0f;
		}
		subpops[i].crowding = 0.0f;
		subpops[i].rank = -1;
		subpops[i].nSelFeatures = 0;
	}

	// Only the parents of each subpopulation are initialized
	for (int it = 0; it < conf -> archipelagoFamilySize; it += conf -> familySize) {
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
 * @brief Evolve one subpopulation running in different modes: Sequential, CPU or GPU only and Heterogeneous (full cooperation between all available devices)
 * @param subpop The subpopulation to evolve
 * @param finished Conditional variable to communicate the handler and the threads that evolve subpopulations
 * @param buffIN Input buffer of individuals that stores individuals from another MPI process
 * @param buffOUT Output buffer of individuals for the subpopulation to evolve
 * @param devicesObject Structure containing the information of a device
 * @param trDataBase The training database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param conf The structure with all configuration parameters
 * @return The number of individuals in the front 0 of the subpopulation
 */
int evolve(Individual *const subpop, bool *const finished, std::vector< Individual > *const buffIN, std::vector< std::vector<Individual> > *const buffOUT, CLDevice *const devicesObject, const float *const trDataBase, const int *const selInstances, const Config *const conf) {


	/********** Multi-objective evaluation over all subpopulations ***********/

	const int nDevices = (omp_get_num_threads() > 1) ? 1 : conf -> nDevices;

	// Auxs
	/*char aux[20];
	sprintf(aux, "pop_%d_%d_%d", nSubpopulations, conf -> subpopulationSize, ind);
	FILE *popData = fopen(aux, "a");*/
	evaluation(subpop, conf -> subpopulationSize, devicesObject, nDevices, trDataBase, selInstances, conf);


	/********** Sort the subpopulation with the 'Non-Domination-Sort' method ***********/

	int nIndsFront0 = nonDominationSort(subpop, conf -> subpopulationSize, conf);
	/*printSubpopulations(subpop, 1, conf);
	fprintf(popData, "%.6g\n", getHypervolume(subpop, nIndsFronts0, conf));*/


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


		/********** Migration with other MPI processes or with the other devices ***********/


		int toPickUp = 0;
		bool stop = false;
		#pragma omp atomic read
		stop = *finished;

		// Provide individuals to the buffer if the communications with the other MPI processes are still open
		if (!stop && conf -> nSubpopulations > 1) {
			//fprintf(stdout, "Evolver %d-%d: ON\n", conf -> mpiRank, omp_get_thread_num());

			// Decide if some individuals are provided to the output buffer and pick up some from either the input buffer or the output buffer of another device
			int threadID = omp_get_thread_num();
			if ((rand() / (float) RAND_MAX) < 0.5f) {

				// Copy some individuals to the ouput buffer
				if (conf -> mpiSize > 1 || (conf -> nIslands > 1 && conf -> nDevices > 1)) {

					#pragma omp critical(buffOUT_access)
					{
						int available = (*buffOUT)[threadID].capacity() - (*buffOUT)[threadID].size();
						if (available > 0) {
							int toCopy = std::max(1, std::min(available, conf -> subpopulationSize >> 2));
							//fprintf(stdout, "Evolver %d-%d: Copying %d individuals to my output buffer\n", conf -> mpiRank, threadID, toCopy);
							(*buffOUT)[threadID].insert((*buffOUT)[threadID].end(), subpop + conf -> subpopulationSize - toCopy, subpop + conf -> subpopulationSize);
						}
						else {
							//fprintf(stdout, "Evolver %d-%d: Output buffer full\n", conf -> mpiRank, threadID);
						}
					}
				}

				// Copy some individuals from the input buffer
				if (conf -> mpiSize > 1) {

					#pragma omp critical(buffIN_access)
					{
						if (!buffIN -> empty() && nChildren < conf -> subpopulationSize) {
							toPickUp = std::min(conf -> subpopulationSize - nChildren, (int) buffIN -> size());
							//fprintf(stdout, "Evolver %d-%d: Copying %d individuals from the input buffer\n", conf -> mpiRank, threadID, toPickUp);
							auto it = buffIN -> rbegin();
							for (int i = 0; i < toPickUp; ++i, ++it) {
								subpop[conf -> subpopulationSize + nChildren + i] = *it;
							}
							buffIN -> erase(buffIN -> end() - toPickUp, buffIN -> end());
						}
						else {
							//fprintf(stdout, "Evolver %d-%d: Input buffer empty\n", conf -> mpiRank, threadID);
						}
					}
				}

				// Copy some individuals from the output buffer of another device
				else if (conf -> nIslands > 1 && conf -> nDevices > 1) {
					int randBuffer = rand() % (*buffOUT).capacity();
					while (randBuffer == threadID) {
						randBuffer = rand() % (*buffOUT).capacity();
					}
					#pragma omp critical(buffOUT_access)
					{
						if (!(*buffOUT)[randBuffer].empty() && nChildren < conf -> subpopulationSize) {
							toPickUp = std::max(1, std::min(conf -> subpopulationSize - nChildren, (int) (*buffOUT)[randBuffer].size() >> 1));
							//fprintf(stdout, "Evolver %d-%d: Copying %d individuals from the output buffer #%d\n", conf -> mpiRank, threadID, toPickUp, randBuffer);
							auto it = (*buffOUT)[randBuffer].rbegin();
							for (int i = 0; i < toPickUp; ++i, ++it) {
								subpop[conf -> subpopulationSize + nChildren + i] = *it;
							}
							(*buffOUT)[randBuffer].erase((*buffOUT)[randBuffer].end() - toPickUp, (*buffOUT)[randBuffer].end());
						}
						else {
							//fprintf(stdout, "Evolver %d-%d: Output buffer #%d empty\n", conf -> mpiRank, threadID, randBuffer);
						}
					}
				}
			}
			else {
				//fprintf(stdout, "Evolver %d-%d: No individuals are provided\n", conf -> mpiRank, threadID);
			}
		}


		// Replace subpopulation
		// Parents and children are sorted by rank and crowding distance.
		// The first "subpopulationSize" individuals will continue for the next generation
		nIndsFront0 = nonDominationSort(subpop, conf -> subpopulationSize + nChildren + toPickUp, conf);
		//printSubpopulations(subpop, 1, conf);


		/********** Get the population quality (calculating the hypervolume) ***********/

		//fprintf(popData, "%.6g\n", getHypervolume(subpop, nIndsFronts0, conf));
	}
	//fclose(popData);
	return nIndsFront0;
}


/**
 * @brief Island-based genetic algorithm model
 * @param subpops The initial subpopulations
 * @param devicesObject Structure containing the information of a device
 * @param trDataBase The training database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param conf The structure with all configuration parameters
 */
void agIslands(Individual *subpops, CLDevice *const devicesObject, const float *const trDataBase, const int *const selInstances, const Config *const conf) {


	/********** MPI variables ***********/

	int array_of_blocklengths[3] = {conf -> nFeatures, conf -> nObjectives + 1, 2};
	MPI::Datatype array_of_types[3] = {MPI::UNSIGNED_CHAR, MPI::FLOAT, MPI::INT};

	// The 'Individual' datatype must be converted to a MPI datatype and commit it
	MPI::Aint array_of_displacement[3] = {offsetof(Individual, chromosome), offsetof(Individual, fitness), offsetof(Individual, rank)};
	MPI::Datatype Individual_MPI_type = MPI::Datatype::Create_struct(3, array_of_blocklengths, array_of_displacement, array_of_types);
	Individual_MPI_type.Commit();

	int color = (conf -> nIslands > 0) ? 1 : 0;
	MPI::Intracomm handlerCOMM = MPI::COMM_WORLD.Split(color, conf -> mpiRank);
	int handlerCOMMRank = handlerCOMM.Get_rank();
	int handlerCOMMSize = handlerCOMM.Get_size();


	/******* Measure and start the algorithm *******/

	MPI::COMM_WORLD.Barrier();
	double timeStart = omp_get_wtime();
	omp_set_nested(1);

	// This MPI process has assigned one subpopulation at least
	if (conf -> nIslands > 0) {

		// Conditional variable to communicate the handler and the threads that evolve subpopulations
		bool finished = false;
		int nIndsFronts0[conf -> nIslands];
		int nEvolvers = std::min(conf -> nDevices, conf -> nIslands);
		int nThreads = (conf -> mpiSize > 1 && conf -> nSubpopulations > 1) ? 2 : 1;

		std::vector< std::vector<Individual> > buffOUT;
		std::vector< Individual> buffIN;
		buffIN.reserve(conf -> subpopulationSize >> 1);
		buffOUT.reserve(nEvolvers);
		for (int b = 0; b < nEvolvers; ++b) {
			std::vector< Individual > aux;
			buffOUT.push_back(aux);
			buffOUT[b].reserve(conf -> subpopulationSize >> 2);
		}

		// Create threads to deal with the subpopulations and one more for the handler if necessary
		#pragma omp parallel num_threads(nThreads)
		{
			// Evolve the subpopulations
			if (omp_get_thread_num() == 0) {

				#pragma omp parallel for schedule(dynamic, 1) num_threads(nEvolvers)
				for (int sp = 0; sp < conf -> nIslands; ++sp) {
					int popIndex = sp * conf -> familySize;
					nIndsFronts0[sp] = evolve(subpops + popIndex, &finished, &buffIN, &buffOUT, &devicesObject[omp_get_thread_num()], trDataBase, selInstances, conf);
				}
				//fprintf(stdout, "Process %d: All subpopulations evolved\n", conf -> mpiRank);

				// Tell the handler to stop because the work has finished
				#pragma omp atomic write
				finished = true;
			}

			// The handler is in charge of receiving and sending individuals to other MPI processes
			else {
				//fprintf(stdout, "Handler %d-%d: Init with size %d\n", conf -> mpiRank, omp_get_thread_num(), handlerCOMMSize);
				std::vector< int > MPItargets(handlerCOMMSize);
				std::iota(MPItargets.begin(), MPItargets.end(), 0);

				// This MPI process will not send to itself
				MPItargets.erase(MPItargets.begin() + handlerCOMMRank);

				bool stop = false;
				Individual *preBuffIN = new Individual[conf -> subpopulationSize >> 2];
				Individual *preBuffOUT = new Individual[conf -> subpopulationSize >> 2];

				// Work while this MPI process is evolving subpopulations and the communications with the other MPI processes are still open
				bool completedIN = true, completedOUT = true;
				MPI::Request requestIN, requestOUT;
				MPI::Status statusIN;
				while (!stop) {

					if (completedIN) {
						requestIN = handlerCOMM.Irecv(preBuffIN, conf -> subpopulationSize >> 2, Individual_MPI_type, MPI::ANY_SOURCE, MPI::ANY_TAG);
						completedIN = false;
						//fprintf(stdout, "Handler %d-%d: Preparing to receive\n", conf -> mpiRank, omp_get_thread_num());
					}

					if (completedOUT) {

						// Check if there are individuals to send in the output buffers
						int randBuffer = rand() % buffOUT.capacity();
						int toPickUp = 0;
						#pragma omp critical(buffOUT_access)
						{

							// Copy the individuals from the output buffer to the pre-buffer (OUT)
							if (!buffOUT[randBuffer].empty()) {
								toPickUp = std::max(1, (int) buffOUT[randBuffer].size() >> 1);
								//fprintf(stdout, "Handler %d-%d: Copying %d individuals from the output buffer #%d\n", conf -> mpiRank, omp_get_thread_num(), toPickUp, randBuffer);
								auto it = buffOUT[randBuffer].rbegin();
								for (int i = 0; i < toPickUp; ++i, ++it) {
									preBuffOUT[i] = *it;
								}
								buffOUT[randBuffer].erase(buffOUT[randBuffer].end() - toPickUp, buffOUT[randBuffer].end());
							}
							else {
								//fprintf(stdout, "Handler %d-%d: Output buffer #%d empty\n", conf -> mpiRank, threadID, randBuffer);
							}
						}

						// If there are individuals in the pre-buffer (OUT) ready to be sent
						if (toPickUp > 0) {
							int MPItarget = MPItargets[rand() % MPItargets.size()];
							requestOUT = handlerCOMM.Isend(preBuffOUT, toPickUp, Individual_MPI_type, MPItarget, MIGRATION);
							completedOUT = false;
							//fprintf(stdout, "Handler %d-%d: Sending %d individuals to MPI process %d\n", conf -> mpiRank, omp_get_thread_num(), toPickUp, MPItarget);
						}
					}

					// Check if the incoming or outgoing message is ready
					completedIN = requestIN.Test(statusIN);
					completedOUT = requestOUT.Test();
					if (completedIN) {

						// Get information about the incoming message
						int source = statusIN.Get_source();
						int tag = statusIN.Get_tag();
						//fprintf(stdout, "Handler %d-%d: Receive the TAG %d from MPI process %d\n", conf -> mpiRank, omp_get_thread_num(), tag, source);

						// Close communications with this MPI process
						if (tag == CLOSE_COMMUNICATIONS) {

							// Remove the MPI process from the target list
							auto it = std::find(MPItargets.begin(), MPItargets.end(), source);
							MPItargets.erase(it);
							//fprintf(stdout, "Handler %d-%d: Closing communication with MPI process %d\n", conf -> mpiRank, omp_get_thread_num(), source);

							// End and tell to the main thread to stop sending individuals to the output buffer
							if (MPItargets.empty()) {
								//fprintf(stdout, "Handler %d-%d: OFF\n", conf -> mpiRank, omp_get_thread_num());
								#pragma omp atomic write
								finished = true;
							}
						}

						// Copy the individuals from the pre-buffer (IN) to the input buffer
						else {
							int count = statusIN.Get_count(Individual_MPI_type);
							#pragma omp critical(buffIN_access)
							{
								int available = buffIN.capacity() - buffIN.size();
								if (available > 0) {
									int toCopy = std::min(available, count);
									//fprintf(stdout, "Handler %d-%d: Copying %d individuals to the input buffer\n", conf -> mpiRank, omp_get_thread_num(), toCopy);
									buffIN.insert(buffIN.end(), preBuffIN, preBuffIN + toCopy);
								}
								else {
									//fprintf(stdout, "Handler %d-%d: Input buffer full\n", conf -> mpiRank, omp_get_thread_num());
								}
							}
						}
					}

					// Check if the handler must finish
					#pragma omp atomic read
					stop = finished;
				}

				delete[] preBuffIN;
				delete[] preBuffOUT;

				// Tell to the other MPI processes with open communication that this MPI process is not available
				if (!MPItargets.empty()) {
					MPI::Request requests[MPItargets.size()];
					for (int pos = 0; pos < MPItargets.size(); ++pos) {
						requests[pos] = handlerCOMM.Isend(NULL, 0, MPI::INT, MPItargets[pos], CLOSE_COMMUNICATIONS);
					}
					//fprintf(stdout, "Handler %d-%d: Waiting to finish the communications\n", conf -> mpiRank, omp_get_thread_num());
					MPI::Request::Waitall(MPItargets.size(), requests);
				}
				else {
					//fprintf(stdout, "Handler %d-%d: No more MPI processes to communicate\n", conf -> mpiRank, omp_get_thread_num());
				}
				//fprintf(stdout, "Handler %d-%d: End\n", conf -> mpiRank, omp_get_thread_num());
			}
		}

		//fprintf(stderr, "Process %d: %.10g\n", conf -> mpiRank, (omp_get_wtime() - timeStart) * 1000.0);


		/********** Recombination process ***********/

		for (int sp = 0; sp < conf -> nIslands; ++sp) {
			//fprintf(stdout, "Process %d: Subpop %d: %.6g %d %d\n", conf -> mpiRank, sp, getHypervolume(subpops + (sp * conf -> familySize), nIndsFronts0[sp], conf), nIndsFronts0[sp], conf -> nIslands);
		}

		if (conf -> mpiRank == 0) {
			Individual *finalSubpop;
			int finalFront0;

			if (conf -> nSubpopulations > 1) {
				if (conf -> mpiSize == 1) {
					finalSubpop = subpops;
				}
				else {
					finalSubpop = new Individual[conf -> totalIndividuals];

					// Each MPI process sends a different number of individuals depending on its number of islands
					int recvCounts[handlerCOMMSize];
					int recvDisplacements[handlerCOMMSize];
					recvCounts[0] = conf -> archipelagoFamilySize;
					recvDisplacements[0] = 0;
					for (int p = 1; p < handlerCOMMSize; ++p) {
						int nIndividuals = (conf -> nSubpopulations / conf -> mpiSize) * conf -> familySize;
						recvCounts[p] = nIndividuals + ((conf -> nSubpopulations % conf -> mpiSize > p) ? conf -> familySize : 0);
						recvDisplacements[p] = recvDisplacements[p - 1] + recvCounts[p - 1];
					}
					/*fprintf(stdout, "[%d", recvCounts[0]);
					for (int p = 1; p < handlerCOMMSize; ++p) {
						fprintf(stdout, ",%d", recvCounts[p]);
					}
					fprintf(stdout, "]\n[%d", recvDisplacements[0]);
					for (int p = 1; p < handlerCOMMSize; ++p) {
						fprintf(stdout, ",%d", recvDisplacements[p]);
					}
					fprintf(stdout, "]\n");*/

					// Collect the subpopulations from each MPI process
					//fprintf(stdout, "Process %d: HERE\n", conf -> mpiRank);
					handlerCOMM.Gatherv(subpops, conf -> archipelagoFamilySize, Individual_MPI_type, finalSubpop, recvCounts, recvDisplacements, Individual_MPI_type, 0);
				}
				//fprintf(stdout, "Process 0: Recombination process\n");
				for (int sp = 0; sp < conf -> nSubpopulations; ++sp) {
					memcpy(finalSubpop + (sp * conf -> subpopulationSize), finalSubpop + (sp * conf -> familySize), conf -> subpopulationSize * sizeof(Individual));
				}

				// The crowding distance of the subpopulation is initialized again for the next nonDominationSort
				#pragma omp parallel for
				for (int i = 0;  i < conf -> worldSize; ++i) {
					finalSubpop[i].crowding = 0.0f;
				}
				finalFront0 = std::min(conf -> subpopulationSize, nonDominationSort(finalSubpop, conf -> worldSize, conf));
			}
			else {
				finalSubpop = subpops;
				finalFront0 = nIndsFronts0[0];
			}

			// All processes must reach this point in order to provide a real time measure
			//fprintf(stdout, "Process %d: HERE\n", conf -> mpiRank);
			handlerCOMM.Barrier();
			//fprintf(stdout, "Process %d: HERE\n", conf -> mpiRank);

			//printSubpopulations(finalSubpop, conf -> nSubpopulations, conf);
			fprintf(stdout, "%.10g\n", (omp_get_wtime() - timeStart) * 1000.0);

			// Get the hypervolume
			fprintf(stdout, "%.6g\n", getHypervolume(finalSubpop, finalFront0, conf));

			// Generation of the Gnuplot file for display the Pareto front
			generateDataPlot(finalSubpop, finalFront0, conf);
			generateGnuplot(conf);

			if (conf -> nSubpopulations > 1 && conf -> mpiSize > 1) {
				delete[] finalSubpop;
			}
		}
		else {
			//fprintf(stdout, "Process %d: HERE %d\n", conf -> mpiRank, conf -> archipelagoFamilySize);
			handlerCOMM.Gatherv(subpops, conf -> archipelagoFamilySize, Individual_MPI_type, NULL, NULL, NULL, NULL, 0);
			//fprintf(stdout, "Process %d: HERE\n", conf -> mpiRank);

			// All processes must reach this point in order to provide a real time measure
			handlerCOMM.Barrier();
		}
	}

	// Variables used are released
	Individual_MPI_type.Free();
	handlerCOMM.Free();
	//fprintf(stdout, "Process %d: OUT\n", conf -> mpiRank);
}
