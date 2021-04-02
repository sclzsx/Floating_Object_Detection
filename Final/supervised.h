#ifndef SUPERVISED_H_
#define SUPERVISED_H_

#include "global_variables.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <map>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using cv::Mat;

void load_images(const string dirname, vector< Mat > &img_lst, bool gray);
//void svm_train(string pos_path, string neg_path, int feature_type, string &xml_filename);

void train(string pos_path, string neg_path, int feature_type, int classifier_type, string &xml_filename);
void ann_classify(Mat input_mat, cv::Rect input_rect, int feature_type, cv::Ptr<cv::ml::ANN_MLP> input_classifier, vector<cv::Rect> &output_rects);
void svm_classify(Mat input_mat, cv::Rect input_rect, int feature_type, cv::Ptr<cv::ml::SVM> input_classifier, vector<cv::Rect> &output_rects);
void knn_classify(Mat input_mat, cv::Rect input_rect, int feature_type, cv::Ptr<cv::ml::KNearest> input_classifier, vector<cv::Rect> &output_rects);
void adaboost_classify(Mat input_mat, cv::Rect input_rect, int feature_type, cv::Ptr<cv::ml::Boost> input_classifier, vector<cv::Rect> &output_rects);
void rtrees_classify(Mat input_mat, cv::Rect input_rect, int feature_type, cv::Ptr<cv::ml::RTrees> input_classifier, vector<cv::Rect> &output_rects);
void bayes_classify(Mat input_mat, cv::Rect input_rect, int feature_type, cv::Ptr<cv::ml::NormalBayesClassifier> input_classifier, vector<cv::Rect> &output_rects);

#endif
