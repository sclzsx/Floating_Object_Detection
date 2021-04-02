/*************************************************
Copyright: Guangyu Zhong all rights reserved
Author: Guangyu Zhong
Date:2015-04-23
Description: codes for Equation 1 in uav saliency (part of the implementation)
Reference http://breckon.eu/toby/publications/papers/sokalski10uavsalient.pdf
**************************************************/
#include "uav.h"

void calculate_contrast(
	const cv::Mat &image,
	const int theta,
	cv::Mat &contrast)
{
	std::vector<cv::Mat> channel;
	cv::split(image, channel);
	channel[0].convertTo(channel[0], CV_32F, 1.0 / 255, 0);
	channel[1].convertTo(channel[1], CV_32F, 1.0 / 255, 0);
	channel[2].convertTo(channel[2], CV_32F, 1.0 / 255, 0);
	int height = image.rows;
	int width = image.cols;
	contrast = cv::Mat(height, width, CV_32F, cv::Scalar::all(0));

	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			std::vector<std::vector<int>> neighbors;
			int length;
			gene_neighbors(i, j, theta, height, width, neighbors, length);
			std::vector<float> distance(length, 0);
			for (int k = 0; k < length; ++k)
			{
				int nei_i = neighbors[k][0];
				int nei_j = neighbors[k][1];
				distance[k] = pow(channel[0].at<float>(i, j) - channel[0].at<float>(nei_i, nei_j), 2) +
					pow(channel[1].at<float>(i, j) - channel[1].at<float>(nei_i, nei_j), 2) +
					pow(channel[2].at<float>(i, j) - channel[2].at<float>(nei_i, nei_j), 2);
				distance[k] = sqrt(distance[k]);
				contrast.at<float>(i, j) += distance[k];
			}
			//std::cout << contrast.at<float>(i, j) << std::endl;
		}
	}
}

void gene_neighbors(const int i, const int j, const int theta, const int height, const int width, std::vector<std::vector<int>> &neighbors, int &length)
{
	std::vector<int> h_range(2);
	std::vector<int> w_range(2);
	h_range[0] = cv::max(0, (i - theta));
	h_range[1] = cv::min(height - 1, (i + theta));
	w_range[0] = cv::max(0, (j - theta));
	w_range[1] = cv::min(width - 1, (j + theta));
	int h_size = h_range[1] - h_range[0] + 1;
	int w_size = w_range[1] - w_range[0] + 1;
	length = h_size*w_size;
	neighbors.resize(length, std::vector<int>(2, 0));
	int ite = 0;
	for (int i = h_range[0]; i <= h_range[1]; ++i)
	{
		for (int j = w_range[0]; j <= w_range[1]; ++j)
		{
			neighbors[ite][0] = i;
			neighbors[ite][1] = j;
			++ite;
		}
	}
}