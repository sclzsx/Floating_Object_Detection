/*************************************************
Copyright: Guangyu Zhong all rights reserved
Author: Guangyu Zhong
Date:2015-04-23
Description: codes for Equation 1 in uav saliency (part of the implementation)
Reference http://breckon.eu/toby/publications/papers/sokalski10uavsalient.pdf
**************************************************/
#ifndef UAV_H_
#define UAV_H_


#define _CRT_SECURE_NO_DEPRECATE 
#include<iostream>

#include<string>
#include <opencv2/opencv.hpp>

void gene_neighbors(
	const int i,
	const int j,
	const int theta,
	const int height,
	const int width,
	std::vector<std::vector<int>> &neighbors,
	int &length);

void calculate_contrast(
	const cv::Mat &image,
	const int theta,
	cv::Mat &contrast);

#endif // uav_H_