#include <curand.h>
#include <curand_kernel.h>

#include "bootstrapKernel.h"

__device__ size_t sampleIDsPitchConst;
__device__ size_t inbagCountsPitchConst;

void setConstantMemoryPitchBootstrap(size_t *sampleIDsPitch, size_t *inbagCountsPitch) {
  cudaMemcpyToSymbol(sampleIDsPitchConst, sampleIDsPitch, sizeof(size_t));
  cudaMemcpyToSymbol(inbagCountsPitchConst, inbagCountsPitch, sizeof(size_t));
}

void setConstantMemoryPitchBootstrap(size_t *sampleIDsPitch)
{
  cudaMemcpyToSymbol(sampleIDsPitchConst, sampleIDsPitch, sizeof(size_t));
}


__global__ void bootstrap_kernel(size_t nTree, size_t nSamples, double sampleFraction, uint seed, size_t* sampleIDs,
    uint* inbagCounts){
  int tid = threadIdx.x;
  uint global_tid = tid + blockIdx.x * blockDim.x;
  int offset = blockDim.x;

  //Row sample
  size_t* rSample = (size_t *)((char *)sampleIDs + blockIdx.x*sampleIDsPitchConst);
  //Row count
  uint* rCount = (uint *)((char *)inbagCounts + blockIdx.x*inbagCountsPitchConst);

  curandState state;
  curand_init(seed, global_tid, 0, &state);
  while(tid < nSamples*sampleFraction){
    uint rand = ceilf(curand_uniform(&state) * nSamples);
    rSample[tid] = rand;
    atomicAdd(&(rCount[rand]), 1);

    tid += offset;
  }
}
