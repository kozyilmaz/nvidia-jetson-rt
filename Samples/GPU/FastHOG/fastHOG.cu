/*
 * fastHog.cpp
 *
 *  Created on: May 14, 2009
 *      Author: viprad
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "HOGEngine.h"
#include "HOGEngineDevice.h"
#include "HOGImage.h"
#include "Others/persondetectorwt.tcc"
extern "C" {
#include "../../gpusync.h"
}

HOGImage image;
cudaStream_t stream;

char file_name[] = "../Samples/GPU/FastHOG/Files/Images/testImage.bmp";

void* Initialize(GPUParameters *parameters) {
  switch (parameters->sync_level) {
  case 0:
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    break;
  case 1:
    cudaSetDeviceFlags(cudaDeviceScheduleYield);
    break;
  case 2:
    cudaSetDeviceFlags(cudaDeviceBlockingSync);
    break;
  default:
    printf("Unknown sync level: %d\n", parameters->sync_level);
    break;
  }
  if (!HOGImageFile(file_name, &image)) {
    printf("Unable to load image file.\n");
    exit(1);
  }
  if (parameters->cuda_device >= 0) {
    if (cudaSetDevice(parameters->cuda_device) != cudaSuccess) {
      printf("Unable to set cuda device.\n");
      exit(1);
    }
  }
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    printf("Unable to create cuda stream.\n");
    exit(1);
  }
  InitializeHOG(image.width, image.height, PERSON_LINEAR_BIAS,
    PERSON_WEIGHT_VEC, PERSON_WEIGHT_VEC_LENGTH);
  return NULL;
}

void MallocCPU(void *thread_data) {
  HostAllocHOGEngineDeviceMemory();
}

void MallocGPU(void *thread_data) {
  DeviceAllocHOGEngineDeviceMemory();
}

void CopyIn(void *thread_data) {
  CopyInHOGEngineDevice();
}

void Exec(void *thread_data) {
  // There are still memcpys to the device in HOGScale and HOGPadding--they
  // may require more work to get rid of because they seem to rely on variables
  // determined during the execution phase.
  BeginProcess(&image, -1, -1, -1, -1, -1.0f, -1.0f);
}

void CopyOut(void *thread_data) {
  EndProcess();
}

void FreeGPU(void *thread_data) {
  DeviceFreeHOGEngineDeviceMemory();
}

void FreeCPU(void *thread_data) {
  HostFreeHOGEngineDeviceMemory();
}

void Finish(void *thread_data) {
  FinalizeHOG();
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  DestroyHOGImage(&image);
  if (cudaDeviceReset() != cudaSuccess) {
    printf("Failed to reset the device.\n");
    exit(1);
  }
}
