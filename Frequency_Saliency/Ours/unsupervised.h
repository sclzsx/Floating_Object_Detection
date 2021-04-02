#ifndef UNSUPERVISED_H_
#define UNSUPERVISED_H_

#include "fft.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <map>

#include "lrr.h"
#include <opencv2/core/eigen.hpp>
#include<numeric>

void MyGammaCorrection(Mat src, Mat& dst, float fGamma);
void get_saliency_rects(Mat input, std::vector<cv::Rect> &output_rects);
void low_rank_decomposition(Mat input, Mat &z, Mat &e);
void get_myphase_saliency2(Mat input, Mat &output);
void get_phase_map(Mat img, Mat &output);

#endif
