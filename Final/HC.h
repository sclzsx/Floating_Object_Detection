#ifndef HC_H_
#define HC_H_

#include "opencv2/core/core.hpp"  
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <map>
#include <functional>
#include "opencv2/imgproc/imgproc_c.h"


class HC
{
public:
	HC()
	{
	}

	~HC()
	{
	}

public:
	void calculateSaliencyMap(cv::Mat src, cv::Mat& dst);
private:
	int Quantize(const cv::Mat& img3f, cv::Mat &idx1i, cv::Mat &_color3f, cv::Mat &_colorNum, double ratio = 0.95);
	void GetHC(const cv::Mat &binColor3f, const cv::Mat &weight1f, cv::Mat &_colorSal);
	void SmoothSaliency(const cv::Mat &binColor3f, cv::Mat &sal1d, float delta, const std::vector<std::vector< std::pair<double, int> >> &similar);
};



#endif // ! HC_H_


