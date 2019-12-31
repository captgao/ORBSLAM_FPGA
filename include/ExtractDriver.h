#ifndef EXTRACTDRIVER_H
#define EXTRACTDRIVER_H

#include <vector>
#include <opencv/cv.h>
namespace ORB_SLAM2{
void FPGAExtract(unsigned char* mat, unsigned int height, unsigned int width, unsigned int batch);

void FPGAResult(std::vector<cv::KeyPoint>& _keypoints,cv::OutputArray& _descriptors,int level);

void FPGACvtMat(cv::Mat cvmat, unsigned char* mat);
}
#endif
