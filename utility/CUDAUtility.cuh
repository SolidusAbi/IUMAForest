/*
 * CUDAUtility.h
 *
 *  Created on: 13/10/2016
 *      Author: abian
 */

#ifndef CUDAUTILITY_H_
#define CUDAUTILITY_H_

// Std Includes
#include <iostream>
#include <vector>

// CUDA Includes
#include <cuda.h>
#include <cuda_runtime.h>

#include "Data.h"

/**
 *	@brief Class that implements all CUDA features
 */
class CUDAUtility {
public:
  static CUDAUtility& getInstance();
  virtual ~CUDAUtility();

  //To eliminate the construct of copy and the assign operator
  CUDAUtility(const CUDAUtility&) = delete;
  CUDAUtility& operator=(const CUDAUtility&) = delete;

  /**
   * @brief Algorithm bootstrap not weighted with replacement for n tree.
   *
   * @param nSamples        number of samples of the training dataset
   * @param sampleFraction  the sample fraction for bootstrap dataset
   * @param nTree           number of tree of the forest
   * @param seeds           seeds per each tree, necessary for the random generator
   * @param samplesIDs      array that contains the samplesIDs per tree
   * @param inbagCounts     histogram of samplesIDs per tree
   */
  void bootstrap(size_t nSamples, double sampleFraction, size_t nTree,
      std::vector<std::vector<size_t>>& samplesIDs, std::vector<std::vector<uint>>& inbagCounts);

  //Test, return elapsed time
  float bootstrapTest(size_t nSamples, double sampleFraction, size_t nTree, std::vector<uint>seeds,
      std::vector<std::vector<size_t>>& samplesIDs, std::vector<std::vector<uint>>& inbagCounts);

  /**
   * @brief Finding the best varID and value for this varID. The best vadID must be one of the
   *  possible varID listed that have been generated randomly.
   *
   * @param data                Dataset for the training of the RF.
   * @param possibleSplitVarIDs vector with the possibles varID generated randomly
   * @param nClasses            number of classes in the training dataset
   * @param nSampleNode         number of samples in the node
   * @param responseClassIDs    vector with the classification result in each sample
   * @param samplesIDsNode      samples used for finding the best split varID
   */
  void findBestSplit(Data *data, std::vector<size_t> possibleSplitVarIDs, size_t nClasses,
      size_t nSampleNode, std::vector<uint> *responseClassIDs, std::vector<size_t> *samplesIDsNode,
      size_t *bestVarID, double *bestValue, double *bestDecrease);

  //Test, return elapsed time
  float findBestSplitTest(Data *data, std::vector<size_t> possibleSplitVarIDs, size_t nClasses,
      size_t nSampleNode, std::vector<uint> *responseClassIDs, std::vector<size_t> *samplesIDsNode,
      size_t *bestVarID, double *bestValue, double *bestDecrease);

//  /**
//   * @brief This function allows you to insert the dataset into the memory of the GPU
//   *
//   * @param data  Contains the dataset
//   * @param nCols number of columns of the dataset
//   * @param nRows number of rows of the dataset
//   */
//  void setDataGPU(double *data, size_t nCols, size_t nRows);

  /**
   * ¡¡¡Chapuza!!! Arreglar!!!
  * @brief This function allows you to insert the dataset into the memory of the GPU
  *
  * @param data  Contains the dataset
  * @param nCols number of columns of the dataset
  * @param nRows number of rows of the dataset
  */
  void setDataGPU(float *data, size_t nCols, size_t nRows);

  /**
   * @brief Free the dataset of the GPU memory
   */
  void freeDataGPU();

  void resetGPU();

  void test(size_t nSamples, std::vector<size_t> *samplesIDsNode, size_t nClasses, std::vector<uint> *responseClassIDs);

private:
  CUDAUtility();


  /**
   * @brief this function convert a pointer of a array to a std::vector<std::vector<T>>
   *
   * @param result  the std::vector<std::vector<T>> result
   * @param array   pointer with the array
   * @param width   Width of matrix set (number of columns)
   * @param height  number of row of the array
   */
  template <typename T>
  void arrayToVector(std::vector<std::vector<T>> &result, T *array, size_t width, size_t height);

  float *dev_data; //chapuza!!
  size_t nCols, nRows;

  cudaDeviceProp deviceProp;
};


#endif /* CUDAUTILITY_H_ */
