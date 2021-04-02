#ifndef _SX_DATA_AUGMENT_H
#define _SX_DATA_AUGMENT_H

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void translation(const cv::Mat src, cv::Mat& dst1, cv::Mat& dst2, cv::Mat& dst3, cv::Mat& dst4, int dx, int dy);

bool AdaptGammaEnhance(cv::Mat Src, cv::Mat &Dst);
bool HistogramEqualization(cv::Mat Src, cv::Mat &equ);
bool LaplacianEnhance(cv::Mat Src, cv::Mat &Lap);
bool LogarithmicTransformation(cv::Mat Src, cv::Mat &imageLog);
bool GammaTransform(cv::Mat Src, cv::Mat &imageGamma);

#endif

