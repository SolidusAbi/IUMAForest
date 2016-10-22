/*
 * CUDAUtility.h
 *
 *  Created on: 13/10/2016
 *      Author: abian
 */

#ifndef CUDAUTILITY_H_
#define CUDAUTILITY_H_

#include <iostream>
#include <vector>

/**
 *	@Brief Class that implements all CUDA features
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
  void bootstrap(int nSamples, double sampleFraction, int nTree, std::vector<uint>seeds,
      std::vector<std::vector<int>>& samplesIDs, std::vector<std::vector<int>>& inbagCounts);

private:
  CUDAUtility();

  /**
   * @brief this function convert a pointer of a array to a std::vector<std::vector<T>>
   *
   * @param result  the std::vector<std::vector<T>> result
   * @param array   pointer with the array
   * @param pitch   pitch without sizeof(T)
   * @param nRow    number of row of the array
   */
  template <typename T>
  void arrayToVector(std::vector<std::vector<T>> &result, T *array, size_t pitch, size_t nRow);

  int maxThreadsPerBlock;
};


#endif /* CUDAUTILITY_H_ */
