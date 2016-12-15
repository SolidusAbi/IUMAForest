#include "findBestSplitKernel.h"
#include <iostream>

__constant__ size_t nPossibleValueMaxConst;
__constant__ size_t dataPitchConst;

void setConstantMemoryPitch(size_t *nPossibleValueMax, size_t *dataPitch) {
  cudaMemcpyToSymbol(nPossibleValueMaxConst, nPossibleValueMax, sizeof(size_t));
  cudaMemcpyToSymbol(dataPitchConst, dataPitch, sizeof(size_t));
}

__global__ void classCountKernel(double *data, size_t nSamples, size_t *samplesID, size_t nClasses, uint *responseClassIDs, size_t *classCounts){
  extern __shared__ size_t classCountTmp[];

  size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
  size_t offset = gridDim.x*blockDim.x;

  if (threadIdx.x < nClasses){
    classCountTmp[threadIdx.x]=0;
  }
  __syncthreads();

  while (tid < nSamples){
    uint classID = responseClassIDs[tid];
    //++classCountTmp[classID];
    atomicAdd((int *)&(classCountTmp[classID]), 1);

    tid+=offset;
  }
  __syncthreads();

  if (threadIdx.x < nClasses){
    atomicAdd((int *)&(classCounts[threadIdx.x]), classCountTmp[threadIdx.x]);
  }

}

__global__ void overallClassCountRightKernel(double* data, size_t nSamples, size_t *samplesID, size_t *possibleSplitVarIDs, size_t *nPossibleValues,
    double *possibleSplitValues, size_t nClasses, size_t* classCounts, uint *responseClassIDs, size_t *nRight, size_t *classCountsRight){

  extern __shared__ size_t countClassTmp[];

  size_t tid = threadIdx.x;
  size_t offset = blockDim.x;
  size_t bid_y = blockIdx.y;
  size_t offsetBlockY = gridDim.y;

  size_t varID = possibleSplitVarIDs[blockIdx.x];
  size_t numSplits = nPossibleValues[blockIdx.x];

  double *possibleSplitValuesVarID = &(possibleSplitValues[blockIdx.x*nPossibleValueMaxConst]);
  size_t *nRightVarID = &(nRight[blockIdx.x*nPossibleValueMaxConst]);
  size_t *classCountsRightVarID = &(classCountsRight[blockIdx.x*nClasses*nPossibleValueMaxConst]);

  size_t *nRightThreadTmp = countClassTmp;
  size_t *classCounRightTmp = &(countClassTmp[blockDim.x]);

  if(tid < nClasses)
    classCounRightTmp[tid]=0;
  __syncthreads();

  while (bid_y < numSplits){
    nRightThreadTmp[threadIdx.x] = 0;

    while (tid < nSamples){
      size_t sampleID = samplesID[tid];
      size_t sampleClassID = responseClassIDs[sampleID];
      //double value = data[varID*nRow + sampleID];
      double value = data[varID*dataPitchConst + sampleID];
      if (value > possibleSplitValuesVarID[bid_y]){
        ++nRightThreadTmp[threadIdx.x];
        atomicAdd((int *)&(classCounRightTmp[sampleClassID]), 1);
      }
      tid += offset;
    }
    __syncthreads();

    if (threadIdx.x == blockDim.x-1){
      size_t nRightTmp = 0;
      for(size_t i = 0; i<blockDim.x; ++i){
        nRightTmp += nRightThreadTmp[i];
      }
      nRightVarID[bid_y] = nRightTmp;
    } else {
      if (threadIdx.x < nClasses)
        classCountsRightVarID[bid_y * nClasses + threadIdx.x] += classCounRightTmp[threadIdx.x];
    }
    __syncthreads();

    if(threadIdx.x < nClasses)
      classCounRightTmp[threadIdx.x]=0;

    tid = threadIdx.x;
    bid_y += offsetBlockY;
  }
}

__global__ void computeDecreaseKernel(size_t nSamples, size_t nClasses, size_t *nPossibleValues, double *possibleSplitValues, size_t* classCounts,
    size_t *classCountsRight, size_t* nRight, double *bestValuePerVarID, double *bestDecreasePerVarID){
  extern __shared__ double tmp[];

  int tid = threadIdx.x;
  int offset = blockDim.x;
  int bid = blockIdx.x;

  size_t *nRightVarID = &(nRight[bid*nPossibleValueMaxConst]);
  size_t *classCountsRightVarID = &(classCountsRight[bid*nClasses*nPossibleValueMaxConst]);
  double *possibleSplitValuesVarID = &(possibleSplitValues[bid*nPossibleValueMaxConst]);

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
