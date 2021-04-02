#ifndef UNSUPERVISED_H_
#define UNSUPERVISED_H_

#include "fft.h"

#include "global_variables.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <map>

#include "BMS.h"
#include "HC.h"
#include "uav.h"

#include "lrr.h"
#include <opencv2/core/eigen.hpp>

void MyGammaCorrection(Mat src, Mat& dst, float fGamma);
void get_sr_saliency(Mat input, Mat &output);
void get_uav_saliency(Mat input, Mat &output);
void get_bms_saliency(Mat input, Mat &output);
void get_hc_saliency(Mat input, Mat &output);

void get_myphase_saliency(Mat input, Mat &output);
void get_saliency_rects(Mat input, int saliency_tpye, vector<cv::Rect> &output_rects);

void low_rank_decomposition(Mat input, Mat &z, Mat &e);
void get_myphase_saliency2(Mat input, Mat &output);

void get_phase_map(Mat input, Mat &output);
#endif
