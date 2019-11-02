#ifndef FINDBESTSPLITKERNEL_H_
#define FINDBESTSPLITKERNEL_H_

void setConstantMemoryPitch(size_t *nPossibleValueMax, size_t *dataPitch);

/**
 * @brief Compute the count samples and generate a overall class counts from all samples
 *
 * @param data              pointer that contains the dataset
 * @param nSamples          number of samples that will be used in the training
 * @param nClasses
 * @param responseClassIDs
 * @param classCounts
 */
__global__ void classCountKernel(float *data, size_t nSamples, size_t *samplesID, size_t nClasses, uint *responseClassIDs, size_t *classCounts);

/**
 * @brief Compute the count samples in right child and generate a overall class counts in right child node.
 *
 * @param data                pointer that contains the dataset
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
__global__ void overallClassCountRightKernel(float* data, size_t nSamples, size_t *samplesID, size_t *possibleSplitVarIDs, size_t *nPossibleValues,
    double *possibleSplitValues, size_t nClasses, size_t* classCounts, uint *responseClassIDs, size_t *nRight, size_t *classCountsRight);

/**
 * @brief computation of decrease for the different varID with different values. It is necessary throw the overallClassCount function.
 *
 * @param nSamples              number of samples that will be used in the training
 * @param nClasses              number of classes of the training dataset
 * @param nPossibleValues       number of possible values per varIDs
 * @param possibleSplitValues   possible values per varIDs
 * @param classCounts           overall class counts in samples
 * @param classCountsRight      overall class counts in right child node (generate with kernel overallClassCountRight())
 * @param nRight                count samples in right child (generate with kernel overallClassCountRight())
 * @param bestValuePerVarID     (output) best value per VarID
 * @param bestDecreasePerVarID  (output) the result of decrease per value of the varID
 */
__global__ void computeDecreaseKernel(size_t nSamples, size_t nClasses, size_t *nPossibleValues, double *possibleSplitValues, size_t* classCounts,
    size_t *classCountsRight, size_t* nRight, double *bestValuePerVarID, double *bestDecreasePerVarID);
#endif /* FINDBESTSPLITKERNEL_H_ */
