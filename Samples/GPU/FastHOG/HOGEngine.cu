#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "HOGDefines.h"
#include "HOGEngine.h"
#include "HOGEngineDevice.h"
#include "HOGImage.h"
#include "HOGResult.h"
#include "HOGUtils.h"

// Contains some global state
struct hog HOG;

extern void SaveResultsToDisk(char* fileName);

void InitializeHOG(int iw, int ih, float svmBias, float* svmWeights,
    int svmWeightsCount) {
  HOG.imageWidth = iw;
  HOG.imageHeight = ih;
  HOG.avSizeX = 48;
  HOG.avSizeY = 96;
  HOG.marginX = 4;
  HOG.marginY = 4;
  HOG.hCellSizeX = 8;
  HOG.hCellSizeY = 8;
  HOG.hBlockSizeX = 2;
  HOG.hBlockSizeY = 2;
  HOG.hWindowSizeX = 64;
  HOG.hWindowSizeY = 128;
  HOG.hNoOfHistogramBins = 9;
  HOG.svmWeightsCount = svmWeightsCount;
  HOG.svmBias = svmBias;
  HOG.svmWeights = svmWeights;
  HOG.wtScale = 2.0f;
  HOG.useGrayscale = false;
  HOG.formattedResultsAvailable = false;
  InitHOG(iw, ih);
}

void FinalizeHOG() {
  CloseHOG();
}

void BeginProcess(HOGImage* hostImage, int _minx, int _miny, int _maxx,
    int _maxy, float minScale, float maxScale) {
  HOG.minX = _minx, HOG.minY = _miny, HOG.maxX = _maxx, HOG.maxY = _maxy;
  if (HOG.minY == -1 && HOG.minY == -1 && HOG.maxX == -1 && HOG.maxY == -1) {
    HOG.minX = 0;
    HOG.minY = 0;
    HOG.maxX = HOG.imageWidth;
    HOG.maxY = HOG.imageHeight;
  }
  BeginHOGProcessing(hostImage->pixels, minScale, maxScale);
}

void EndProcess() {
  HOG.cppResult = EndHOGProcessing();
  GetHOGParameters();
  ComputeFormattedResults();
  // printf("Found %d positive results.\n", HOG.formattedResultsCount);
  // SaveResultsToDisk(file_name);
}

void SaveResultsToDisk(char* fileName) {
  FILE* f;
  f = fopen(fileName, "w+");
  if (!f) {
    printf("Error! Failed opening output file!\n");
    return;
  }
  fprintf(f, "%d\n", HOG.formattedResultsCount);
  for (int i = 0; i < HOG.formattedResultsCount; i++) {
    fprintf(f, "%f %f %d %d %d %d %d %d\n",
      HOG.formattedResults[i].scale, HOG.formattedResults[i].score,
      HOG.formattedResults[i].width, HOG.formattedResults[i].height,
      HOG.formattedResults[i].x, HOG.formattedResults[i].y,
      HOG.formattedResults[i].origX, HOG.formattedResults[i].origY);
  }
  fclose(f);
}

void ComputeFormattedResults() {
  int i, j, k, resultId;
  int leftoverX, leftoverY, currentWidth, currentHeight, rNumberOfWindowsX,
    rNumberOfWindowsY;
  resultId = 0;
  HOG.formattedResultsCount = 0;
  float* currentScaleWOffset;
  float currentScale = HOG.startScale;
  for (i = 0; i < HOG.scaleCount; i++) {
    currentScaleWOffset = HOG.cppResult + i * HOG.hNumberOfWindowsX *
      HOG.hNumberOfWindowsY;
    for (j = 0; j < HOG.hNumberOfWindowsY; j++) {
      for (k = 0; k < HOG.hNumberOfWindowsX; k++) {
        float score = currentScaleWOffset[k + j * HOG.hNumberOfWindowsX];
        if (score > 0) HOG.formattedResultsCount++;
      }
    }
  }

  for (i = 0; (i < HOG.scaleCount) && (resultId < MAX_RESULTS); i++) {
    currentScaleWOffset = HOG.cppResult + i * HOG.hNumberOfWindowsX *
      HOG.hNumberOfWindowsY;
    for (j = 0; j < HOG.hNumberOfWindowsY; j++) {
      for (k = 0; k < HOG.hNumberOfWindowsX; k++) {
        float score = currentScaleWOffset[k + j * HOG.hNumberOfWindowsX];
        if (score <= 0) continue;
        currentWidth = iDivUpF(HOG.hPaddedWidth, currentScale);
        currentHeight = iDivUpF(HOG.hPaddedHeight, currentScale);
        rNumberOfWindowsX = (currentWidth - HOG.hWindowSizeX) /
          HOG.hCellSizeX + 1;
        rNumberOfWindowsY = (currentHeight - HOG.hWindowSizeY) /
          HOG.hCellSizeY + 1;
        leftoverX = (currentWidth - HOG.hWindowSizeX - HOG.hCellSizeX *
          (rNumberOfWindowsX - 1)) / 2;
        leftoverY = (currentHeight - HOG.hWindowSizeY - HOG.hCellSizeY *
          (rNumberOfWindowsY - 1)) / 2;
        HOG.formattedResults[resultId].origX = k * HOG.hCellSizeX + leftoverX;
        HOG.formattedResults[resultId].origY = j * HOG.hCellSizeY + leftoverY;
        HOG.formattedResults[resultId].width = (int) floorf(
          (float) HOG.hWindowSizeX * currentScale);
        HOG.formattedResults[resultId].height = (int) floorf(
          (float) HOG.hWindowSizeY * currentScale);
        HOG.formattedResults[resultId].x = (int)ceilf(currentScale *
          (HOG.formattedResults[resultId].origX + HOG.hWindowSizeX / 2) -
          (float) HOG.hWindowSizeX * currentScale / 2) - HOG.hPaddingSizeX +
          HOG.minX;
        HOG.formattedResults[resultId].y = (int)ceilf(currentScale *
          (HOG.formattedResults[resultId].origY + HOG.hWindowSizeY / 2) -
          (float) HOG.hWindowSizeY * currentScale / 2) - HOG.hPaddingSizeY +
          HOG.minY;
        HOG.formattedResults[resultId].scale = currentScale;
        HOG.formattedResults[resultId].score = score;
        resultId++;
      }
    }
    currentScale = currentScale * HOG.scaleRatio;
  }
}
