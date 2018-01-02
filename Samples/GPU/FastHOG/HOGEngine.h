#ifndef HOG_ENGINE_H
#define HOG_ENGINE_H
#include "HOGDefines.h"
#include "HOGImage.h"
#include "HOGResult.h"

struct hog {
  int imageWidth, imageHeight;
  int avSizeX, avSizeY, marginX, marginY;
  int scaleCount;
  int hCellSizeX, hCellSizeY;
  int hBlockSizeX, hBlockSizeY;
  int hWindowSizeX, hWindowSizeY;
  int hNoOfHistogramBins;
  int hPaddedWidth, hPaddedHeight;
  int hPaddingSizeX, hPaddingSizeY;
  int minX, minY, maxX, maxY;
  float wtScale;
  float startScale, endScale, scaleRatio;
  int svmWeightsCount;
  float svmBias, *svmWeights;
  int hNoOfCellsX, hNoOfCellsY;
  int hNoOfBlocksX, hNoOfBlocksY;
  int hNumberOfWindowsX, hNumberOfWindowsY;
  int hNumberOfBlockPerWindowX, hNumberOfBlockPerWindowY;
  bool useGrayscale;
  float* cppResult;
  HOGResult formattedResults[MAX_RESULTS];
  bool formattedResultsAvailable;
  int formattedResultsCount;
};

extern struct hog HOG;

extern cudaStream_t stream;

void InitializeHOG(int iw, int ih, float svmBias, float* svmWeights,
    int svmWeightsCount);

void FinalizeHOG();

void BeginProcess(HOGImage* hostImage,
    int _minx, int _miny, int _maxx, int _maxy, float minScale,
    float maxScale);

void EndProcess();

void ComputeFormattedResults();

#endif // HOG_ENGINE_H
