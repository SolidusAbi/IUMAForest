#ifndef BOOTSTRAPKERNEL_H_
#define BOOTSTRAPKERNEL_H_

  void setConstantMemoryPitchBootstrap(size_t *sampleIDsPitch, size_t *inbagCountsPitch);

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
  __global__ void bootstrap_kernel(size_t nTree, size_t nSamples, double sampleFraction,
      uint* seed, size_t* sampleIDs, uint* inbagCounts);

#endif
