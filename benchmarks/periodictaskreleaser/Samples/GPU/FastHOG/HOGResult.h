#ifndef __HOG_RESULT__
#define __HOG_RESULT__

typedef struct hogresult {
  float score;
  float scale;
  int width;
  int height;
  int origX;
  int origY;
  int x;
  int y;
} HOGResult;

#endif

