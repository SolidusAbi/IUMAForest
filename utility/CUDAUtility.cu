/*
 * CUDAUtility.cpp
 *
 *  Created on: 13/10/2016
 *      Author: abian
 */

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#include <random>

#include "CUDAUtility.cuh"

#define min(a,b) (a<b?a:b);

__device__ size_t sampleIDsPitch_const;
__device__ size_t inbagCountsPitch_const;

/**
 * @brief Algorithm bootstrap not weighted with replacement for n tree. It will generate N bootstrap samples with the objective to generate a bootstrap sample for echa tree.
 *
 * @param nTree number of bootstrap samples that it have to generate.
 * @param nSamples number of samples of the original dataset.
 * @param sampleFraction number of fraction for each sample.
 * @param seed a array with random seed for each tree for to generate random number. The seed length is equal to nTree.
 * @param sampleIDs (output) bootstrap sample for each tree.
 * @param inbagCounts (output) histogram of each bootstrap sample.
 */
__global__ void bootstrap_kernel(size_t nTree, size_t nSamples, double sampleFraction, uint* seed, size_t* sampleIDs,
    uint* inbagCounts){
  int tid = threadIdx.x;
  int offset = blockDim.x;

  /*Generating a random number in a specific ranger:
    1- Use CURAND to generate a uniform distribution between 0.0 and 1.0
    2- Then multiply this by the desired range (largest value - smallest value + 0.999999).
    3- Then add the offset (+ smallest value).
    4- Then truncate to an integer.
  */
  curandState state;
  curand_init(seed[blockIdx.x], tid + blockIdx.x * blockDim.x, 0, &state);
  while(tid < nSamples*sampleFraction){
    float randf = curand_uniform(&state);
    randf *= ((nSamples - 1) - 0) + 0.999999;
    randf += 0;
    int rand = (int)truncf(randf);

    //Row sample
    size_t* rSample = (size_t *)((char *)sampleIDs + blockIdx.x*sampleIDsPitch_const);
    rSample[tid] = rand;

    //Row count
    uint* rCount = (uint *)((char *)inbagCounts + blockIdx.x*inbagCountsPitch_const);
    atomicAdd(&(rCount[rand]), 1);

    tid += offset;
  }
}

CUDAUtility::CUDAUtility() : maxThreadsPerBlock(512) {}

CUDAUtility::~CUDAUtility() {}

CUDAUtility& CUDAUtility::getInstance(){
	static CUDAUtility instance;
	return instance;
}

void CUDAUtility::bootstrap(size_t nSamples, double sampleFraction, size_t nTree, std::vector<uint>seeds,
    std::vector<std::vector<size_t>>& samplesIDs, std::vector<std::vector<uint>>& inbagCounts){

  //Host var
  size_t *host_sampleIDs;
  uint *host_inbagCounts;
  host_sampleIDs = (size_t *)malloc((int)(nSamples * sampleFraction * nTree) * sizeof(size_t));
  host_inbagCounts = (uint *)malloc(nSamples * nTree * sizeof(int));
  //How i use memory in 2D, I need the pitch
  size_t host_sampleIDs_pitch = nSamples * sampleFraction * sizeof(size_t);
  size_t host_inbagCounts_pitch = nSamples * sizeof(int);

  //Device var
  size_t *dev_sampleIDs;
  uint *dev_inbagCounts, *dev_seed;
  size_t dev_sampleIDs_pitch, dev_inbagCounts_pitch;

  cudaMallocPitch((void **)&dev_sampleIDs, &dev_sampleIDs_pitch, (int)(nSamples * sampleFraction) * sizeof(size_t), nTree);
  cudaMallocPitch((void **)&dev_inbagCounts, &dev_inbagCounts_pitch, nSamples * sizeof(int), nTree);
  cudaMalloc((void **)&dev_seed, nTree * sizeof(int));
  cudaMemcpy(dev_seed, seeds.data(), nTree * sizeof(int), cudaMemcpyHostToDevice);

  //Initialize the histogram of inbag samples
  cudaMemset2D(dev_inbagCounts, dev_inbagCounts_pitch, 0, nSamples * sizeof(int), nTree);

  cudaMemcpyToSymbol(sampleIDsPitch_const, &dev_sampleIDs_pitch, sizeof(size_t));
  cudaMemcpyToSymbol(inbagCountsPitch_const, &dev_inbagCounts_pitch, sizeof(size_t));

  int threadsPerBlock = min(nSamples, maxThreadsPerBlock);
  bootstrap_kernel<<<nTree,threadsPerBlock>>>(nTree, nSamples, sampleFraction, dev_seed, dev_sampleIDs,
      dev_inbagCounts);

  cudaMemcpy2D(host_sampleIDs, host_sampleIDs_pitch, dev_sampleIDs, dev_sampleIDs_pitch, host_sampleIDs_pitch,
      nTree, cudaMemcpyDeviceToHost);
  cudaMemcpy2D(host_inbagCounts, host_inbagCounts_pitch, dev_inbagCounts, dev_inbagCounts_pitch,
      host_inbagCounts_pitch, nTree, cudaMemcpyDeviceToHost);

  arrayToVector(samplesIDs, host_sampleIDs, host_sampleIDs_pitch/sizeof(size_t), nTree);
  arrayToVector(inbagCounts, host_inbagCounts, host_inbagCounts_pitch/sizeof(int), nTree);

  free(host_sampleIDs);
  free(host_inbagCounts);
  cudaFree(dev_sampleIDs);
  cudaFree(dev_inbagCounts);
  cudaFree(dev_seed);

  return;
}

template<typename T>
void CUDAUtility::arrayToVector(std::vector<std::vector<T>> &result, T *array, size_t width, size_t height){

  for (int i=0; i<height; ++i){
    std::vector<T> row ( &array[i*width], &array[(i+1)*width] );
    result.push_back(row);
  }

  return;
}
