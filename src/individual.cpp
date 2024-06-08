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
 * @file individual.cpp
 * @author Juan José Escobar Pérez
 * @date 26/06/2015
 * @brief Implementation of the non-dominated sorting method
 * @copyright Hpmoon (c) 2015 University of Granada
 */

/********************************* Includes *******************************/

#include "individual.h"
#include <algorithm> // sort...
#include <math.h> // INFINITY...
#include <vector> // std::vector...

/********************************* Methods ********************************/

/**
 * @brief Perform non-dominated sorting on the subpopulation
 * @param subpop Current subpopulation
 * @param nIndividuals The number of individuals which will be sorted
 * @param conf The structure with all configuration parameters
 * @return The number of individuals in the front 0
 */
int nonDominationSort(Individual *const subpop, const int nIndividuals, const Config *const conf) {

	// Individuals to classify (really they are not 'Individual' objects)
	// Each 'Individual' p of 'indivDomination' contains the number of individuals...
	// ...who dominate p and a list of individuals who are dominated by p
	std::vector< std::pair< std::vector<int>, int> > indivDomination(nIndividuals);
	std::vector< std::vector<int> > front(1);

	// Search for individuals who belong to the first front
	int nFronts = 0;
	for (int i = 0; i < nIndividuals; ++i) {
		for (int j = i + 1; j < nIndividuals; ++j) {
			u_char domLess = 0;
			u_char domEqual = 0;
			u_char domMore = 0;
			for (u_char obj = 0; obj < conf -> nObjectives; ++obj) {
				if (subpop[i].fitness[obj] < subpop[j].fitness[obj]) {
					++domLess;
				}
				else if (subpop[i].fitness[obj] == subpop[j].fitness[obj]) {
					++domEqual;
				}
				else {
					++domMore;
				}
			}

			if (domLess == 0 && domEqual != conf -> nObjectives) {
				++indivDomination[i].second;
				indivDomination[j].first.push_back(i);
			}
			else if (domMore == 0 && domEqual != conf -> nObjectives) {
				indivDomination[i].first.push_back(j);
				++indivDomination[j].second;
			}
		}

		if (indivDomination[i].second == 0) {
			subpop[i].rank = 0;
			front[0].push_back(i);
		}
	}

	// Find the subsequent fronts
	while (!front[nFronts].empty()) {
		front.push_back(std::vector<int>());
		int sizeActualFront = front[nFronts].size();
		for (int i = 0; i < sizeActualFront; ++i) {
			int nDomByInd = (int) indivDomination[front[nFronts][i]].first.size();
			for (int j = 0; j < nDomByInd; ++j) {
				int dominateToInd = indivDomination[front[nFronts][i]].first[j];
				int nDomToInd = (--indivDomination[dominateToInd].second);
				if (nDomToInd == 0) {
					subpop[dominateToInd].rank = nFronts + 1;
						front[nFronts + 1].push_back(dominateToInd);
				}
			}
		}
		++nFronts;
	}

	front.pop_back();

	// Sort the individuals according to the rank
	std::sort(subpop, subpop + nIndividuals, rankCompare());

	// Find the crowding distance for each individual in each front
	for (int f = 0, i = 0; f < nFronts; ++f) {
		int sizeFrontI = (int) front[f].size();
		Individual *begin = subpop + i;
		Individual *end = begin + sizeFrontI;
		for (u_char obj = 0; obj < conf -> nObjectives; ++obj) {
			std::sort(begin, end, objectiveCompare(obj));
			float fMin = begin -> fitness[obj];
			float fMax = (end - 1) -> fitness[obj];
			begin -> crowding = INFINITY;
			(end - 1) -> crowding = INFINITY;
			bool fMaxFminZero = (fMax == fMin);

			for (int j = 1; j < sizeFrontI - 1; ++j) {
				Individual *current = begin + j;
				if (fMaxFminZero) {
					current -> crowding = INFINITY;
				}
				else if (current -> crowding != INFINITY) {
					float nextObj = (current + 1) -> fitness[obj];
					float previousObj = (current - 1) -> fitness[obj];
					current -> crowding += (nextObj - previousObj) / (fMax - fMin);
				}
			}
		}
		
		i += sizeFrontI;
	}

	// Sort the individuals according to the rank and Crowding distance
	std::sort(subpop, subpop + nIndividuals, rankAndCrowdingCompare());

	return front[0].size();
}
