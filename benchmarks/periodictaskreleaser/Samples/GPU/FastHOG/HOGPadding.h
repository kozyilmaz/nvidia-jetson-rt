#ifndef __HOG_PADDING__
#define __HOG_PADDING__
#include "HOGDefines.h"

void InitPadding();
void ClosePadding();

void PadHostImage(uchar4* registeredImage, float4 *paddedRegisteredImage,
      int minx, int miny, int maxx, int maxy);

void DeviceAllocHOGPaddingMemory(void);
void DeviceFreeHOGPaddingMemory(void);

#endif
