/*
* HOGImage.h
*
*  Created on: May 14, 2009
*      Author: viprad
*/
#ifndef __HOGIMAGE_H__
#define __HOGIMAGE_H__
extern enum ImageType {
 IMAGE_RESIZED,
 IMAGE_COLOR_GRADIENTS,
 IMAGE_GRADIENT_ORIENTATIONS,
 IMAGE_PADDED,
 IMAGE_ROI,
} imagetype;


typedef struct hogimage {
  bool isLoaded;
  int width, height;
  unsigned char* pixels;
} HOGImage;

bool HOGImageFile(const char *fileName, HOGImage *image);

void DestroyHOGImage(HOGImage *image);

#endif /* HOGIMAGE_H_ */
