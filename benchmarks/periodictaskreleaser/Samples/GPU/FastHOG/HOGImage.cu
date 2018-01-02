/*
 * HOGImage.cpp
 *
 *  Created on: May 14, 2009
 *      Author: viprad
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <FreeImage.h>
#include "HOGImage.h"

// Loads the given image file into the HOGImage struct. Returns false on error.
bool HOGImageFile(const char* fileName, HOGImage *image) {
  int bpp;
  FIBITMAP *bmp = 0;
  FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
  fif = FreeImage_GetFileType(fileName);
  if (fif == FIF_UNKNOWN) {
    fif = FreeImage_GetFIFFromFilename(fileName);
  }
  if ((fif == FIF_UNKNOWN) || !FreeImage_FIFSupportsReading(fif)) {
    image->isLoaded = false;
    return false;
  }
  bmp = FreeImage_Load(fif, fileName, 0);
  if (!bmp) {
    image->isLoaded = false;
    return false;
  }
  image->width = FreeImage_GetWidth(bmp);
  image->height = FreeImage_GetHeight(bmp);
  bpp = FreeImage_GetBPP(bmp);
  if (bpp != 32) {
    FIBITMAP *bmpTemp = FreeImage_ConvertTo32Bits(bmp);
    FreeImage_Unload(bmp);
    if (!bmpTemp) {
      image->isLoaded = false;
      return false;
    }
    bmp = bmpTemp;
    bpp = FreeImage_GetBPP(bmp);
  }
  image->pixels = (unsigned char*) malloc(sizeof(unsigned char) * 4 *
    image->width * image->height);
  FreeImage_ConvertToRawBits(image->pixels, bmp, image->width * 4, bpp,
    FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, true);
  FreeImage_Unload(bmp);
  image->isLoaded = true;
  return true;
}

void DestroyHOGImage(HOGImage *image) {
  free(image->pixels);
  image->pixels = NULL;
  image->isLoaded = false;
}
