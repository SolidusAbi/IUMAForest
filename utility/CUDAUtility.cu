/*
 * CUDAUtility.cpp
 *
 *  Created on: 13/10/2016
 *      Author: abian
 */

// Std Includes
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <random>
#include <cmath>
#include <iostream>

// CUDA Includes
#include <curand_kernel.h>
#include <device_launch_parameters.h>

// Project Includes
#include "CUDAUtility.cuh"
#include "kernels/bootstrapKernel.h"
#include "kernels/findBestSplitKernel.h"

#include "Timer.h"


#define min(a,b) (a<b?a:b);

CUDAUtility::CUDAUtility() : dev_data(nullptr), nCols(0), nRows(0), availableMemory(0)  {
  cudaDeviceReset();
  cudaGetDeviceProperties(&deviceProp, 0);
  availableMemory = deviceProp.totalGlobalMem * 0.8;
}

CUDAUtility::~CUDAUtility() {
  cudaFree(dev_data);
}

CUDAUtility& CUDAUtility::getInstance(){
	static CUDAUtility instance;
	return instance;
}

void CUDAUtility::bootstrap(size_t nSamples, double sampleFraction, size_t nTree,
    std::vector<std::vector<size_t>>& samplesIDs, std::vector<std::vector<uint>>& inbagCounts){

  //Host var
  size_t *host_sampleIDs;
  uint *host_inbagCounts;
  host_sampleIDs = (size_t *)malloc((int)(nSamples * sampleFraction * nTree) * sizeof(size_t));
  host_inbagCounts = (uint *)malloc(nSamples * nTree * sizeof(int));
  //As I use memory in 2D, I need the pitch
  size_t host_sampleIDs_pitch = nSamples * sampleFraction * sizeof(size_t);
  size_t host_inbagCounts_pitch = nSamples * sizeof(int);

  //Device var
  size_t *dev_sampleIDs;
  //uint *dev_inbagCounts;
  size_t dev_sampleIDs_pitch, dev_inbagCounts_pitch;

  // (int)(nSamples * sampleFraction) * sizeof(size_t)
  // nSamples * sizeof(int)

  cudaMallocPitch((void **)&dev_sampleIDs, &dev_sampleIDs_pitch, (int)(nSamples * sampleFraction) * sizeof(size_t), nTree);
  //cudaMallocPitch((void **)&dev_inbagCounts, &dev_inbagCounts_pitch, nSamples * sizeof(int), nTree);

  //Initialize the histogram of inbag samples
  //cudaMemset2D(dev_inbagCounts, dev_inbagCounts_pitch, 0, nSamples * sizeof(int), nTree);

  setConstantMemoryPitchBootstrap(&dev_sampleIDs_pitch, nullptr);

  int threadsPerBlock = min(nSamples, deviceProp.maxThreadsPerBlock);
//  bootstrap_kernel<<<nTree,threadsPerBlock>>>(nTree, nSamples, sampleFraction, time(0), dev_sampleIDs,
//     dev_inbagCounts);
  bootstrap_kernel<<<nTree,threadsPerBlock>>>(nTree, nSamples, sampleFraction, time(0), dev_sampleIDs);

  cudaMemcpy2D(host_sampleIDs, host_sampleIDs_pitch, dev_sampleIDs, dev_sampleIDs_pitch, host_sampleIDs_pitch,
      nTree, cudaMemcpyDeviceToHost);
//  cudaMemcpy2D(host_inbagCounts, host_inbagCounts_pitch, dev_inbagCounts, dev_inbagCounts_pitch,
//     host_inbagCounts_pitch, nTree, cudaMemcpyDeviceToHost);

  arrayToVector(samplesIDs, host_sampleIDs, host_sampleIDs_pitch/sizeof(size_t), nTree);
//  arrayToVector(inbagCounts, host_inbagCounts, host_inbagCounts_pitch/sizeof(int), nTree);


  free(host_sampleIDs);
  free(host_inbagCounts);
  cudaFree(dev_sampleIDs);

//  cudaFree(dev_inbagCounts);
}

float CUDAUtility::bootstrapTest(size_t nSamples, double sampleFraction, size_t nTree, std::vector<uint>seeds,
    std::vector<std::vector<size_t>>& samplesIDs, std::vector<std::vector<uint>>& inbagCounts){

  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  cudaEventRecord(startEvent, 0);

  bootstrap(nSamples, sampleFraction, nTree, samplesIDs, inbagCounts);

  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  float elapsedTimeCuda;
  cudaEventElapsedTime(&elapsedTimeCuda, startEvent, stopEvent);

  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);

  return ((float)elapsedTimeCuda/1000);
}

void CUDAUtility::findBestSplit(Data *data, std::vector<size_t> possibleSplitVarIDs, size_t nClasses,
    size_t nSampleNode, std::vector<uint> *responseClassIDs, std::vector<size_t> *samplesIDsNode,
    size_t *bestVarID, double *bestValue, double *bestDecrease){

  *bestVarID = 0;
  *bestDecrease = -1;
  *bestValue = 0;

  //getting the possible split values per varID, to facilitate work on the GPU
  std::vector<std::vector<double>> possibleSplitValues;
  std::vector<size_t> nPossibleSplitValues;
  size_t nPossibleSplitValuesMax = 0;
  for(size_t i=0; i<possibleSplitVarIDs.size(); ++i){
    std::vector<double> possiblesValues;
    data->getAllValues(possiblesValues, *samplesIDsNode, possibleSplitVarIDs[i]);
    // Remove largest value because no split possible
    //possiblesValues.pop_back();

    nPossibleSplitValues.push_back(possiblesValues.size());
    possibleSplitValues.push_back(possiblesValues);
    if(nPossibleSplitValuesMax < possiblesValues.size())
      nPossibleSplitValuesMax =  possiblesValues.size();
  }

  setConstantMemoryPitch(&nPossibleSplitValuesMax, &(this->nRows));

  //Prepare the possible values data for introduce in the GPU
  double *possiblesSplitValuesPerVarID = (double *)malloc(possibleSplitVarIDs.size() * nPossibleSplitValuesMax * sizeof(double));
  for (size_t i=0; i<possibleSplitVarIDs.size(); ++i){
    if (possibleSplitValues[i].size() != nPossibleSplitValuesMax){
      //Fill elements to form the array
      while (possibleSplitValues[i].size() != nPossibleSplitValuesMax){
        possibleSplitValues[i].push_back(std::numeric_limits<double>::infinity());
      }
    }

    for (size_t j=0; j<nPossibleSplitValuesMax; ++j)
      possiblesSplitValuesPerVarID[i*nPossibleSplitValuesMax + j] = possibleSplitValues[i][j];
  }

  size_t *dev_possibleSplitVarIDs;
  cudaMalloc((void **)& dev_possibleSplitVarIDs, possibleSplitVarIDs.size()*sizeof(size_t));
  cudaMemcpy(dev_possibleSplitVarIDs, possibleSplitVarIDs.data(), possibleSplitVarIDs.size()*sizeof(size_t), cudaMemcpyHostToDevice);

  //Introduce the number of possible values per varID
  size_t *dev_nPossibleSplitValuesPerVarID;
  cudaMalloc((void **)& dev_nPossibleSplitValuesPerVarID, possibleSplitVarIDs.size()*sizeof(size_t));
  cudaMemcpy(dev_nPossibleSplitValuesPerVarID, nPossibleSplitValues.data(), possibleSplitVarIDs.size()*sizeof(size_t), cudaMemcpyHostToDevice);

  double *dev_possibleSplitValuesPerVarID;
  cudaMalloc((void **)& dev_possibleSplitValuesPerVarID, possibleSplitVarIDs.size() * nPossibleSplitValuesMax * sizeof(double));
  cudaMemcpy(dev_possibleSplitValuesPerVarID, possiblesSplitValuesPerVarID, possibleSplitVarIDs.size() * nPossibleSplitValuesMax * sizeof(double),
      cudaMemcpyHostToDevice);

  size_t *dev_samplesID;
  cudaMalloc((void **)& dev_samplesID, nSampleNode*sizeof(size_t));
  cudaMemcpy(dev_samplesID, (*samplesIDsNode).data(), nSampleNode*sizeof(size_t), cudaMemcpyHostToDevice);

  uint *dev_responseClassID;
  cudaMalloc((void **)& dev_responseClassID, (*responseClassIDs).size()*sizeof(int));
  cudaMemcpy(dev_responseClassID, (*responseClassIDs).data(), (*responseClassIDs).size()*sizeof(int), cudaMemcpyHostToDevice);

  //Introduce the class counts generate in CPU
  size_t *dev_classCounts;
  cudaMalloc((void **)& dev_classCounts, nClasses*sizeof(size_t));
  cudaMemset(dev_classCounts, 0, nClasses*sizeof(size_t));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  uint maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
  size_t nBlocks = truncf(nSampleNode/(maxThreadsPerBlock*2));
  classCountKernel<<<nBlocks, maxThreadsPerBlock, nClasses*sizeof(size_t)>>>(dev_data, nSampleNode, dev_samplesID, nClasses, dev_responseClassID, dev_classCounts);

  size_t *dev_nRight;
  cudaMalloc((void **)& dev_nRight, possibleSplitVarIDs.size()*nPossibleSplitValuesMax*sizeof(size_t));
  cudaMemset(dev_nRight, 0, possibleSplitVarIDs.size()*nPossibleSplitValuesMax*sizeof(size_t));

  //Preallocate memory on GPU that will be used for internal process. Preallocate has a better perfomance than dynamic allocate.
  size_t *dev_classCountsRight;
  cudaMalloc((void **)& dev_classCountsRight, possibleSplitVarIDs.size()*nPossibleSplitValuesMax*nClasses*sizeof(size_t));
  cudaMemset(dev_classCountsRight, 0, possibleSplitVarIDs.size()*nPossibleSplitValuesMax*nClasses*sizeof(size_t));

  dim3 grid(possibleSplitVarIDs.size(),prop.multiProcessorCount * 4);
  int threadsPerBlock = min(nSampleNode, maxThreadsPerBlock);
  overallClassCountRightKernel<<<grid,threadsPerBlock,(threadsPerBlock+nClasses)*sizeof(size_t)>>>(this->dev_data, nSampleNode, dev_samplesID, dev_possibleSplitVarIDs,
      dev_nPossibleSplitValuesPerVarID, dev_possibleSplitValuesPerVarID, nClasses, dev_classCounts, dev_responseClassID, dev_nRight, dev_classCountsRight);

  cudaFree(dev_samplesID);
  cudaFree(dev_possibleSplitVarIDs);
  cudaFree(dev_responseClassID);

  double *dev_bestValuePerVarID, *dev_bestDecreasePerVarID;
  cudaMalloc((void **)& dev_bestValuePerVarID, possibleSplitVarIDs.size()*sizeof(double));
  cudaMalloc((void **)& dev_bestDecreasePerVarID, possibleSplitVarIDs.size()*sizeof(double));

  computeDecreaseKernel<<<possibleSplitVarIDs.size(),threadsPerBlock,2*threadsPerBlock*sizeof(double)>>>(nSampleNode, nClasses, dev_nPossibleSplitValuesPerVarID,
      dev_possibleSplitValuesPerVarID, dev_classCounts, dev_classCountsRight, dev_nRight, dev_bestValuePerVarID, dev_bestDecreasePerVarID);

  double *bestValuePerVarID = (double *)malloc(possibleSplitVarIDs.size()*sizeof(double));
  double *bestDecreasePerVarID = (double *)malloc(possibleSplitVarIDs.size()*sizeof(double));
  cudaMemcpy(bestDecreasePerVarID, dev_bestDecreasePerVarID, possibleSplitVarIDs.size()*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(bestValuePerVarID, dev_bestValuePerVarID, possibleSplitVarIDs.size()*sizeof(double), cudaMemcpyDeviceToHost);

  for (size_t i=0; i<possibleSplitVarIDs.size(); ++i){
    if (bestDecreasePerVarID[i] > *bestDecrease){
      *bestVarID = possibleSplitVarIDs[i];
      *bestDecrease = bestDecreasePerVarID[i];
      *bestValue = bestValuePerVarID[i];
    }
  }

  cudaFree(dev_nPossibleSplitValuesPerVarID);
  cudaFree(dev_possibleSplitValuesPerVarID);
  cudaFree(dev_classCounts);
  cudaFree(dev_nRight);
  cudaFree(dev_classCountsRight);
  cudaFree(dev_bestValuePerVarID);
  cudaFree(dev_bestDecreasePerVarID);

  free(bestValuePerVarID);
  free(bestDecreasePerVarID);
}

float CUDAUtility::findBestSplitTest(Data *data, std::vector<size_t> possibleSplitVarIDs, size_t nClasses,
    size_t nSampleNode, std::vector<uint> *responseClassIDs, std::vector<size_t> *samplesIDsNode,
    size_t *bestVarID, double *bestValue, double *bestDecrease){

  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  cudaEventRecord(startEvent, 0);

  findBestSplit(data, possibleSplitVarIDs, nClasses, nSampleNode, responseClassIDs, samplesIDsNode, bestVarID, bestValue, bestDecrease);

  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  float elapsedTimeCuda;
  cudaEventElapsedTime(&elapsedTimeCuda, startEvent, stopEvent);

  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);

  //printf("Resultado en GPU...\n");
  //printf("Time for operation: %3.6f seconds\n", elapsedTimeCuda/1000);

  return elapsedTimeCuda/1000;
}

template<typename T>
void CUDAUtility::arrayToVector(std::vector<std::vector<T>> &result, T *array, size_t width, size_t height){

  for (int i=0; i<height; ++i){
    std::vector<T> row ( &array[i*width], &array[(i+1)*width] );
    result.push_back(row);
  }

  return;
}

//void CUDAUtility::setDataGPU(double *data, size_t nCols, size_t nRows){
//  this->nCols = nCols;
//  this->nRows = nRows;
//  cudaMalloc((void **)&dev_data, nCols * nRows * sizeof(double));
//  cudaMemcpy(dev_data, data, nCols * nRows * sizeof(double), cudaMemcpyHostToDevice);
//}

void CUDAUtility::setDataGPU(float *data, size_t nCols, size_t nRows){
  this->nCols = nCols;
  this->nRows = nRows;
  cudaMalloc((void **)&dev_data, nCols * nRows * sizeof(float));
  cudaMemcpy(dev_data, data, nCols * nRows * sizeof(float), cudaMemcpyHostToDevice);
}

void CUDAUtility::freeDataGPU(){
  cudaFree(dev_data);
}

void CUDAUtility::resetGPU(){
  cudaDeviceReset();
}
