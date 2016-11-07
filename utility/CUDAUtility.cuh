/*
 * CUDAUtility.h
 *
 *  Created on: 13/10/2016
 *      Author: Abian Hernandez
 */

#ifndef CUDAUTILITY_H_
#define CUDAUTILITY_H_

#include <iostream>
#include <vector>

#include "DataDouble.h"

/**
 * @brief Class that implements all CUDA features
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
  void bootstrap(size_t nSamples, double sampleFraction, size_t nTree, std::vector<uint>seeds,
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
  void findBestSplit(DataDouble *data, std::vector<size_t> possibleSplitVarIDs, size_t nClasses,
      size_t nSampleNode, std::vector<uint> *responseClassIDs, std::vector<size_t> *samplesIDsNode,
      size_t *bestVarID, double *bestValue, double *bestDecrease);

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

  int maxThreadsPerBlock;
};


#endif /* CUDAUTILITY_H_ */
