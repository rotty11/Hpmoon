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
 * @file clUtils.h
 * @author Juan José Escobar Pérez
 * @date 01/07/2015
 * @brief Function declarations of the OpenCL device manager
 * @copyright Hpmoon (c) 2015 University of Granada
 */

#ifndef CLUTILS_H
#define CLUTILS_H

/********************************* Includes *******************************/

#include "individual.h" // Individual
#include <CL/cl.h> // OpenCL
#include <vector> // std::vector...

/******************************** Constants *******************************/

const char *const CL_ERROR_PLATFORMS_NUMBER = "Error: Could not get the number of platforms";
const char *const CL_ERROR_PLATFORMS_FOUND = "Error: Platforms not found";
const char *const CL_ERROR_PLATFORM_ID = "Error: Could not get the platform id";
const char *const CL_ERROR_DEVICES_NUMBER = "Error: Could not get the number of devices";
const char *const CL_ERROR_DEVICE_ID = "Error: Could not get the device id";
const char *const CL_ERROR_DEVICE_NAME = "Error: Could not get the device name";
const char *const CL_ERROR_DEVICE_MAXCU = "Error: Could not get maximum number of compute units";
const char *const CL_ERROR_DEVICE_MAXWORKITEMS = "Error: Could not get maximum number of work-items in each dimension";
const char *const CL_ERROR_DEVICE_TYPE = "Error: Could not get the device type";
const char *const CL_ERROR_DEVICE_MAXMEM = "Error: Could not get the maximum local memory of the device";
const char *const CL_ERROR_DEVICE_LOCALMEM = "Error: Local memory exceeded";
const char *const CL_ERROR_DEVICE_CONTEXT = "Error: Could not get the context";
const char *const CL_ERROR_DEVICE_QUEUE = "Error: Could not get the command queue";
const char *const CL_ERROR_FILE_OPEN = "Error: An error ocurred opening the kernel file";
const char *const CL_ERROR_PROGRAM_BUILD = "Error: Could not create the program";
const char *const CL_ERROR_PROGRAM_ERRORS = "Error: Could not get the compilation errors";
const char *const CL_ERROR_KERNEL_BUILD = "Error: Could not create the kernel";
const char *const CL_ERROR_OBJECT_SUBPOPS = "Error: Could not create the OpenCL object containing the subpopulations";
const char *const CL_ERROR_OBJECT_TRDB = "Error: Could not create the OpenCL object containing the training database";
const char *const CL_ERROR_OBJECT_CENTROIDS = "Error: Could not create the OpenCL object containing the indexes of the initial centroids";
const char *const CL_ERROR_KERNEL_ARGUMENT1 = "Error: Could not set the first kernel argument";
const char *const CL_ERROR_KERNEL_ARGUMENT2 = "Error: Could not set the second kernel argument";
const char *const CL_ERROR_KERNEL_ARGUMENT3 = "Error: Could not set the third kernel argument";
const char *const CL_ERROR_ENQUEUE_TRDB = "Error: Could not enqueue the OpenCL object containing the training database";
const char *const CL_ERROR_ENQUEUE_CENTROIDS = "Error: Could not enqueue the OpenCL object containing the init centroids";
const char *const CL_ERROR_OBJECT_TTRDB = "Error: Could not create the OpenCL object containing the transposed training database";
const char *const CL_ERROR_ENQUEUE_TTRDB = "Error: Could not enqueue the OpenCL object containing the transposed training database";
const char *const CL_ERROR_KERNEL_ARGUMENT6 = "Error: Could not set the sixth kernel argument";
const char *const CL_ERROR_DEVICE_FOUND = "Error: Not exists the specified device";

/********************************* Structures ********************************/

/**
 * @brief Structure containing the OpenCL variables of a device
 */
typedef struct CLDevice {


	/**
	 * @brief Identifier for the device
	 */
	cl_device_id device;


	/**
	 * @brief The device type
	 */
	cl_device_type deviceType;


	/**
	 * @brief The context associated to the device
	 */
	cl_context context;


	/**
	 * @brief The command queue which contains the tasks (reads/writes on the device...)
	 */
	cl_command_queue commandQueue;


	/**
	 * @brief The OpenCL kernel with the implementation of K-means
	 */
	cl_kernel kernel;


	/**
	 * @brief OpenCL object which contains the training database
	 */
	cl_mem objTrDataBase;


	/**
	 * @brief OpenCL object which contains the indexes of the instances choosen as initial centroids
	 */
	cl_mem objSelInstances;


	/**
	 * @brief OpenCL object which contains the current subpopulations
	 */
	cl_mem objSubpopulations;


	/**
	 * @brief OpenCL object which contains the transposed training database
	 */
	cl_mem objTransposedTrDataBase;


	/**
	 * @brief The number of compute units specified for this device
	 */
	int computeUnits;


	/**
	 * @brief The number of global work-items specified for this device
	 */
	size_t wiGlobal;


	/**
	 * @brief The number of local work-items specified for this device
	 */
	size_t wiLocal;


	/**
	 * @brief The device name
	 */
	std::string deviceName;


	/********************************* Methods ********************************/

	/**
	 * @brief The destructor
	 */
	~CLDevice();

} CLDevice;

/********************************* Methods ********************************/

/**
 * @brief Creates an array of objects containing the OpenCL variables of each device
 * @param trDataBase The training database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param transposedTrDataBase The training database already transposed
 * @param conf The structure with all configuration parameters
 * @return A pointer containing the objects
 */
CLDevice *createDevices(const float *const trDataBase, const int *const selInstances, const float *const transposedTrDataBase, Config *const conf);


/**
 * @brief Gets the IDs of all available OpenCL devices
 * @return A vector containing the IDs of all devices
 */
std::vector<cl_device_id> getAllDevices();


/**
 * @brief Prints a list containing the ID of all available OpenCL devices
 * @param mpiRank The MPI process number which is calling the function
 */
void listDevices(const int mpiRank);

#endif
