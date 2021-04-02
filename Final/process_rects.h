#ifndef PROCESS_RECTS_H_
#define PROCESS_RECTS_H_

#include "global_variables.h"

#include <dlib/data_io.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <map>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using cv::Mat;

void dlibRectangleToOpenCV(dlib::rectangle r, cv::Rect &output);
void get_groundtruth_rects(string inputfile_filename, vector<vector<cv::Rect>> &groundtruths);
void test(vector<cv::Rect> input_rects, int frame_num, vector<cv::Rect> &output_rects);
void get_nms_rects(vector<cv::Rect> input_rects, int frame_num, vector<cv::Rect> &output_rects);
void combine_two_methods_rects(vector<cv::Rect> time_rects, vector<cv::Rect> frequency_rects, vector<cv::Rect> &output_rects, float th1, float th2);
void get_confidences_of_rects(vector<cv::Rect> input_rects, int frame_num, vector<float> &confidences);
void get_correct_rects(vector<cv::Rect> input_rects, vector<cv::Rect> thisframe_groundtruths, int thresh, vector<cv::Rect> &output_rects);
void get_mid_rect(vector<cv::Rect> input_rects, cv::Rect &output_rect);
void dbscan_process(vector<cv::Rect> input_rects, vector<cv::Rect> &output_rects);
void show_rects(string winname, cv::Mat input, vector<cv::Rect> input_rects);
void mergeRects(vector<cv::Rect> rects, cv::Rect &output);
float getDistance(cv::Point A, cv::Point B);
void compare_two_rects(cv::Rect rectA, cv::Rect rectB, int &flag);

#endif