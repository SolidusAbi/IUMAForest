#include <curand.h>
#include <curand_kernel.h>

#include "bootstrapKernel.h"

__device__ size_t sampleIDsPitchConst;
__device__ size_t inbagCountsPitchConst;

void setConstantMemoryPitchBootstrap(size_t *sampleIDsPitch, size_t *inbagCountsPitch) {
  cudaMemcpyToSymbol(sampleIDsPitchConst, sampleIDsPitch, sizeof(size_t));
  cudaMemcpyToSymbol(inbagCountsPitchConst, inbagCountsPitch, sizeof(size_t));
}


__global__ void bootstrap_kernel(size_t nTree, size_t nSamples, double sampleFraction, uint* seed, size_t* sampleIDs,
    uint* inbagCounts){
  int tid = threadIdx.x;
  int offset = blockDim.x;

  /**Generating a random number in a specific ranger:
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
    size_t* rSample = (size_t *)((char *)sampleIDs + blockIdx.x*sampleIDsPitchConst);
    rSample[tid] = rand;

    //Row count
    uint* rCount = (uint *)((char *)inbagCounts + blockIdx.x*inbagCountsPitchConst);
    atomicAdd(&(rCount[rand]), 1);

    tid += offset;
  }
}
