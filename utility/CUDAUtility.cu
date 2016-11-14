/*
 * CUDAUtility.cpp
 *
 *  Created on: 13/10/2016
 *      Author: abian
 */

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>

#include <random>
#include <limits>

#include "CUDAUtility.cuh"

#define min(a,b) (a<b?a:b);

__device__ size_t sampleIDsPitch_const;
__device__ size_t inbagCountsPitch_const;

__device__ size_t nPossibleValueMax_Const;

/**
 * @brief Algorithm bootstrap not weighted with replacement for n tree. It will generate N bootstrap samples with the objective
 *  to generate a bootstrap sample for each tree.
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

/**
 * @brief Compute the count samples in right child and generate a overall class counts in right child node.
 *
 * @param data                pointer that contains the dataset
 * @param nCol                number of columns of the dataset
 * @param nRow                number of rows of the dataset
 * @param samplesID           IDs of the samples that will be used in the training
 * @param nSamples            number of samples that will be used in the training
 * @param possibleSplitVarIDs IDs of the attributes that could be used in the node
 * @param nPossibleValues     number of possible values per varIDs
 * @param possibleSplitValues possible values per varIDs
 * @param nClasses            number of classes of the training dataset
 * @param responseClassIDs    ID of the classification in each sample
 * @param classCounts         overall class counts in samples per node
 * @param classCountsRight    (result) overall class counts in right child node
 * @param nRight              (result) count samples in right child
 */
__global__ void overallClassCountRight (double* data, size_t nCol, size_t nRow, size_t *samplesID, size_t nSamples, size_t *possibleSplitVarIDs, size_t *nPossibleValues,
    double *possibleSplitValues, size_t nClasses, uint *responseClassIDs, size_t* classCounts, size_t *classCountsRight, size_t* nRight){

  int tid = threadIdx.x;
  int offset = blockDim.x;
  int bid_y = blockIdx.y;
  int offsetBlockY = gridDim.y;

  size_t varID = possibleSplitVarIDs[blockIdx.x];
  size_t numSplits = nPossibleValues[blockIdx.x];

  size_t *nRightVarID = &(nRight[blockIdx.x*nPossibleValueMax_Const]);
  size_t *classCountsRightVarID = &(classCountsRight[blockIdx.x*nClasses*nPossibleValueMax_Const]);
  double *possibleSplitValuesVarID = &(possibleSplitValues[blockIdx.x*nPossibleValueMax_Const]);

  //Count samples in right child per class and possbile split
  while(bid_y < nSamples){
    size_t sampleID = samplesID[bid_y];
    double value = data[varID*nRow + sampleID]; //ok
    uint sample_classID = responseClassIDs[sampleID]; //ok

    while(tid < numSplits){
      if (value > possibleSplitValuesVarID[tid]){
        atomicAdd((int *)&(nRightVarID[tid]), 1);
        atomicAdd((int *)&(classCountsRightVarID[tid * nClasses + sample_classID]), 1);
      }

      tid += offset;
    }
    bid_y += offsetBlockY;
    tid = threadIdx.x;
  }
}

/**
 * @brief computation of decrease for the different varID with different values. It is necessary throw the overallClassCount function.
 *
 * @param nSamples              number of samples that will be used in the training
 * @param nClasses              number of classes of the training dataset
 * @param nPossibleValues       number of possible values per varIDs
 * @param possibleSplitValues   possible values per varIDs
 * @param classCounts           overall class counts in samples
 * @param classCountsRight      overall class counts in right child node
 * @param nRight                count samples in right child
 * @param bestValuePerVarID     (output) best value per VarID
 * @param bestDecreasePerVarID  (output) the result of decrease per value of the varID
 */
__global__ void computeDecrease(size_t nSamples, size_t nClasses, size_t *nPossibleValues, double *possibleSplitValues, size_t* classCounts,
    size_t *classCountsRight, size_t* nRight, double *bestValuePerVarID, double *bestDecreasePerVarID){
  extern __shared__ double tmp[];

  int tid = threadIdx.x;
  int offset = blockDim.x;
  int bid = blockIdx.x;

  size_t *nRightVarID = &(nRight[bid*nPossibleValueMax_Const]);
  size_t *classCountsRightVarID = &(classCountsRight[bid*nClasses*nPossibleValueMax_Const]);
  double *possibleSplitValuesVarID = &(possibleSplitValues[bid*nPossibleValueMax_Const]);

  double *bestDecreasePerThread = tmp;
  double *bestValuePerThread = &(tmp[blockDim.x]);

  bestDecreasePerThread[threadIdx.x]=-1;
  size_t nSplits = nPossibleValues[bid];

  while (tid < nSplits){
    size_t nLeft = nSamples - nRightVarID[tid];
    if (nLeft != 0 && nRightVarID[tid] != 0) {
      //Sum of squares
      double sumLeft = 0;
      double sumRight = 0;

      for (size_t i = 0; i < nClasses; ++i) {
        size_t class_count_right = classCountsRightVarID[tid * nClasses + i];
        size_t class_count_left = classCounts[i] - class_count_right;

        sumRight += class_count_right * class_count_right;
        sumLeft += class_count_left * class_count_left;
      }

      //Decrease of impurity
      double decrease = sumLeft / (double) nLeft + sumRight / (double) nRightVarID[tid];

      if(decrease > bestDecreasePerThread[threadIdx.x]){
        bestValuePerThread[threadIdx.x] = possibleSplitValuesVarID[tid];
        bestDecreasePerThread[threadIdx.x] = decrease;
      }
    }

    tid += offset;
  }
  __syncthreads();

  if (threadIdx.x == 0){
    double bestDecrease = -1;
    double bestValue = 0;
    for (size_t i=0; i<blockDim.x; ++i){
      if (bestDecreasePerThread[i] > bestDecrease){
        bestDecrease = bestDecreasePerThread[i];
        bestValue = bestValuePerThread[i];
      }
    }
    bestValuePerVarID[blockIdx.x] = bestValue;
    bestDecreasePerVarID[blockIdx.x] = bestDecrease;
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

void CUDAUtility::findBestSplit(DataDouble *data, std::vector<size_t> possibleSplitVarIDs, size_t nClasses,
    size_t nSampleNode, std::vector<uint> *responseClassIDs, std::vector<size_t> *samplesIDsNode,
    size_t *bestVarID, double *bestValue, double *bestDecrease){

  *bestVarID = 0;
  *bestDecrease = -1;
  *bestValue = 0;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  cudaEventRecord(startEvent, 0);

  size_t* classCounts = new size_t[nClasses]();
  // Compute overall class counts
  for (size_t i = 0; i < nSampleNode; ++i) {
    size_t sampleID = (*samplesIDsNode)[i];
    uint sample_classID = (*responseClassIDs)[sampleID];
    ++classCounts[sample_classID];
  }

  //getting the possible split values per varID, to facilitate work on the GPU
  std::vector<std::vector<double>> possibleSplitValues;
  std::vector<size_t> nPossibleSplitValues;
  size_t nPossibleSplitValuesMax = 0;
  for(size_t i=0; i<possibleSplitVarIDs.size(); ++i){
    std::vector<double> possiblesValues;
    data->getAllValues(possiblesValues, *samplesIDsNode, possibleSplitVarIDs[i]);
    // Remove largest value because no split possible
    possiblesValues.pop_back();

    nPossibleSplitValues.push_back(possiblesValues.size());
    possibleSplitValues.push_back(possiblesValues);
    if(nPossibleSplitValuesMax < possiblesValues.size())
      nPossibleSplitValuesMax =  possiblesValues.size();
  }

  cudaMemcpyToSymbol(nPossibleValueMax_Const, &nPossibleSplitValuesMax, sizeof(size_t));

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

  //Introduce the dataset in GPUMemory
  double *dev_data;
  cudaMalloc((void **)&dev_data, data->getNumCols() * data->getNumRows() * sizeof(double));
  cudaMemcpy(dev_data, data->getData(), data->getNumCols() * data->getNumRows() * sizeof(double), cudaMemcpyHostToDevice);

  //Introduce the number of possible values per varID
  size_t *dev_nPossibleSplitValuesPerVarID;
  cudaMalloc((void **)& dev_nPossibleSplitValuesPerVarID, possibleSplitVarIDs.size()*sizeof(size_t));
  cudaMemcpy(dev_nPossibleSplitValuesPerVarID, nPossibleSplitValues.data(), possibleSplitVarIDs.size()*sizeof(size_t), cudaMemcpyHostToDevice);

  size_t *dev_samplesID;
  cudaMalloc((void **)& dev_samplesID, nSampleNode*sizeof(size_t));
  cudaMemcpy(dev_samplesID, (*samplesIDsNode).data(), nSampleNode*sizeof(size_t), cudaMemcpyHostToDevice);

  //Introducir los posibles varID
  size_t *dev_possibleSplitVarIDs;
  cudaMalloc((void **)& dev_possibleSplitVarIDs, possibleSplitVarIDs.size()*sizeof(size_t));
  cudaMemcpy(dev_possibleSplitVarIDs, possibleSplitVarIDs.data(), possibleSplitVarIDs.size()*sizeof(size_t), cudaMemcpyHostToDevice);

  //Introducir los posibles values
  double *dev_possibleSplitValuesPerVarID;
  cudaMalloc((void **)& dev_possibleSplitValuesPerVarID, possibleSplitVarIDs.size() * nPossibleSplitValuesMax * sizeof(double));
  cudaMemcpy(dev_possibleSplitValuesPerVarID, possiblesSplitValuesPerVarID, possibleSplitVarIDs.size() * nPossibleSplitValuesMax * sizeof(double),
      cudaMemcpyHostToDevice);

  //Introduce the class counts generate in CPU
  size_t *dev_classCounts;
  cudaMalloc((void **)& dev_classCounts, nClasses*sizeof(size_t));
  cudaMemcpy(dev_classCounts, classCounts, nClasses*sizeof(size_t), cudaMemcpyHostToDevice);

  //Preallocate memory on GPU that will be used for internal process. Preallocate has a better perfomance than dynamic allocate.
  size_t *dev_ClassCountsRight;
  cudaMalloc((void **)& dev_ClassCountsRight, possibleSplitVarIDs.size()*nPossibleSplitValuesMax*nClasses*sizeof(size_t));
  cudaMemset(dev_ClassCountsRight, 0, possibleSplitVarIDs.size()*nPossibleSplitValuesMax*nClasses*sizeof(size_t));

  size_t *dev_nRight;
  cudaMalloc((void **)& dev_nRight, possibleSplitVarIDs.size()*nPossibleSplitValuesMax*sizeof(size_t));
  cudaMemset(dev_nRight, 0, possibleSplitVarIDs.size()*nPossibleSplitValuesMax*sizeof(size_t));

  uint *dev_responseClassID;
  cudaMalloc((void **)& dev_responseClassID, (*responseClassIDs).size()*nClasses*sizeof(int));
  cudaMemcpy(dev_responseClassID, (*responseClassIDs).data(), (*responseClassIDs).size()*nClasses*sizeof(int), cudaMemcpyHostToDevice);

  double *dev_bestValuePerVarID, *dev_bestDecreasePerVarID;
  cudaMalloc((void **)& dev_bestValuePerVarID, possibleSplitVarIDs.size()*sizeof(double));
  cudaMalloc((void **)& dev_bestDecreasePerVarID, possibleSplitVarIDs.size()*sizeof(double));

  dim3 grid(possibleSplitVarIDs.size(),prop.multiProcessorCount * 4);
  int threadsPerBlock = min(nPossibleSplitValuesMax, maxThreadsPerBlock);
  overallClassCountRight<<<grid,threadsPerBlock>>>(dev_data, data->getNumCols(), data->getNumRows(), dev_samplesID, nSampleNode, dev_possibleSplitVarIDs,
      dev_nPossibleSplitValuesPerVarID, dev_possibleSplitValuesPerVarID, nClasses, dev_responseClassID, dev_classCounts, dev_ClassCountsRight,
      dev_nRight);

  cudaFree(dev_data);
  cudaFree(dev_samplesID);
  cudaFree(dev_possibleSplitVarIDs);
  cudaFree(dev_responseClassID);

  computeDecrease<<<possibleSplitVarIDs.size(),threadsPerBlock,2*threadsPerBlock*sizeof(double)>>>(nSampleNode, nClasses, dev_nPossibleSplitValuesPerVarID,
      dev_possibleSplitValuesPerVarID, dev_classCounts, dev_ClassCountsRight, dev_nRight, dev_bestValuePerVarID, dev_bestDecreasePerVarID);

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

  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  float elapsedTimeCuda;
  cudaEventElapsedTime(&elapsedTimeCuda, startEvent, stopEvent);

  free(possiblesSplitValuesPerVarID);
  free(classCounts);

  cudaFree(dev_nPossibleSplitValuesPerVarID);
  cudaFree(dev_possibleSplitValuesPerVarID);
  cudaFree(dev_classCounts);
  cudaFree(dev_ClassCountsRight);
  cudaFree(dev_nRight);
  cudaFree(dev_bestValuePerVarID);
  cudaFree(dev_bestDecreasePerVarID);

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
