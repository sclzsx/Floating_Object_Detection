#pragma once

#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/objdetect/objdetect.hpp>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>

using std::to_string;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using cv::Mat;


void GetPath(string PosPath, string NegPath, string namepos, string nameneg, vector<string> &posSamplesPath, vector<string> &negSamplesPath);
void my_feature(Mat resized_gray, vector<float> &descriptors);
void my_feature2(Mat resized_gray, Mat &feature);
void GetFeatureAndLabel(string PosPath, string NegPath, string namepos, string nameneg, Mat &FeatureMat, Mat &LabelMat);
void GetFeatureMat(vector<Mat> inputMats, Mat &FeatureMat);
void load_images(const string dirname, vector< Mat > &img_lst, bool gray);

void save_data(string dataname, string labelname, string inputpath1, string inputpath2);
void train();
void accuracy();
void save();
float getDistance(cv::Point A, cv::Point B);
bool rectA_intersect_rectB(cv::Rect rectA, cv::Rect rectB);
void MyGammaCorrection(Mat src, Mat& dst, float fGamma);
Mat get_sr(Mat I);
cv::Rect mergeRects(vector<cv::Rect> rects);
void detect();
