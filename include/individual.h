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
 * @file individual.h
 * @author Juan José Escobar Pérez
 * @date 26/06/2015
 * @brief Function declarations of the non-dominated sorting method
 * @copyright Hpmoon (c) 2015 University of Granada
 */

#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

/********************************** Includes *********************************/

#include "config.h" // 'Config' datatype

/********************************* Structures ********************************/

/**
 * @brief Structure containing the Individual's parameters
 */
typedef struct Individual {


	/**
	 * @brief Vector denoting the selected features
	 *
	 * Values: Zeros or ones
	 */
	unsigned char chromosome[N_FEATURES];


	/**
	 * @brief Individual fitness for the multi-objective functions
	 *
	 * Values: Each position contains an objective function
	 */
	float fitness[2];


	/**
	 * @brief Crowding distance of the individual
	 *
	 * The values are positives or infinites
	 */
	float crowding;


	/**
	 * @brief Range of the individual (Pareto front)
	 */
	int rank;


	/**
	 * @brief Number of selected features
	 */
	int nSelFeatures;

} Individual;


/**
 * @brief Structure that contains a comparator to sort individuals according to their ranks
 */
struct rankCompare {


	/**
	 * @brief Compare individuals according to their ranks
	 * @param ind1 The first individual
	 * @param ind2 The second individual
	 * @return true if the rank of the first individual is lower than the rank of the second individual
	 */
	bool operator ()(const Individual &ind1, const Individual &ind2) const {
		return ind1.rank < ind2.rank;
	}
};


/**
 * @brief Structure that contains a function to sort individuals by fitness (objective function)
 * 
 * The attribute 'objective' specifies which objective function should be compared
 */
struct objectiveCompare {


	/**
	 * @brief The objective function which should be compared
	 */
	unsigned char objective;


	/**
	 * @brief Constructor
	 * @param objective The objective function which should be compared
	 */
	objectiveCompare(unsigned char objective) {
		this -> objective = objective;
	}


	/**
	 * @brief Compare individuals according to their objectives
	 * @param ind1 The first individual
	 * @param ind2 The second individual
	 * @return true if the fitness of the first individual is lower than the fitness of the second individual
	 */
	bool operator ()(const Individual &ind1, const Individual &ind2) const {
		return ind1.fitness[this -> objective] < ind2.fitness[this -> objective];
	}
};


/**
 * @brief Structure that contains a function to sort individuals by rank and crowding distance
 * 
 * If both individuals have the same rank, the crowding distance will be compared
 */
struct rankAndCrowdingCompare {


	/**
	 * @brief Compare individuals according to their ranks and their crowding distances
	 * @param ind1 The first individual
	 * @param ind2 The second individual
	 * @return true if the rank of the first individual is lower than the rank of the second individual. If both individuals have the same rank, the crowding distance will be compared
	 */
	bool operator ()(const Individual &ind1, const Individual &ind2) const {
		if (ind1.rank == ind2.rank) {
			return ind1.crowding > ind2.crowding;
		}
		else {
			return ind1.rank < ind2.rank;
		}
	}
};

/********************************* Methods ********************************/

/**
 * @brief Perform non-dominated sorting on the subpopulation
 * @param subpop Current subpopulation
 * @param nIndividuals The number of individuals which will be sorted
 * @param conf The structure with all configuration parameters
 * @return The number of individuals in the front 0
 */
int nonDominationSort(Individual *const subpop, const int nIndividuals, const Config *const conf);

#endif
