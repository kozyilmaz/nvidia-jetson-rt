#ifndef __HOG_SVM_SLIDER__
#define __HOG_SVM_SLIDER__
#include "HOGDefines.h"

void InitSVM();
void CloseSVM();
void ResetSVMScores(float1* svmScores);
void LinearSVMEvaluation(float1* svmScores, float1* blockHistograms, int noHistogramBins,
          int windowSizeX, int windowSizeY,
                                  int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY,
          int hogBlockCountX, int hogBlockCountY,
          int scaleId, int width, int height);

__global__ void linearSVMEvaluation(float1* svmScores, float svmBias,
            float1* blockHistograms, int noHistogramBins,
            int windowSizeX, int windowSizeY, int hogBlockCountX, int hogBlockCountY,
            int cellSizeX, int cellSizeY,
            int numberOfBlockPerWindowX, int numberOfBlockPerWindowY,
            int blockSizeX, int blockSizeY, int alignedBlockDimX,
            int scaleId, int scaleCount,
            int hNumberOfWindowsX, int hNumberOfWindowsY, int width, int height);

void DeviceAllocHOGSVMMemory(void);
void CopyInHOGSVM(void);
void DeviceFreeHOGSVMMemory(void);
#endif
