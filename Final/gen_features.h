#ifndef GEN_FEATURES_H_
#define GEN_FEATURES_H_

#include "global_variables.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <map>
#include <numeric>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using cv::Mat;

void get_hog_feature_vector(Mat input, vector<float> &output);

void get_fhog_feature_vector(Mat input, vector<float> &output);

void feature_vec2mat(vector<float> input, Mat &output);

void ComputeLBPImage_Uniform(const Mat &srcImage, Mat &LBPImage);

void ComputeLBPFeatureVector_Uniform(const Mat &srcImage, cv::Size cellSize, Mat &featureVector);

void get_lbp_feature_vector(Mat input, vector<float> &output);

void get_glcm_feature_vector(Mat input, vector<float> &output);

void get_fhog_glcm_feature_vector(Mat input, vector<float> &output);

#endif
