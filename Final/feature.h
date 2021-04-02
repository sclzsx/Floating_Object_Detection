#ifndef FEATURE_H_
#define FEATURE_H_

#include <opencv2/opencv.hpp>
#include <iostream>
//#include <algorithm>
//#include <vector>
//#include <string>
//#include <cmath>
#include <fstream>
#include <io.h>
#include "HalconCpp.h"
#include "HDevThread.h"

using namespace HalconCpp;

namespace IPSG
{
	class Feature
	{
	public:
		Feature(int _flag = 0, float _radius = 0.0) :flag(_flag), radius(_radius) {}
		~Feature()
		{
		
		}
		bool   abstractSampleFeature(std::string PsamplePath, int PsampleLabel, std::string NsamplePath, int NsampleLabel,std::string savePath);
		void   getFile(std::string path, std::vector<std::string>& files);
		bool   mHuMoment(std::string inputImageFile, std::vector<double>& vecFeature);
		bool   abstractFeature(cv::Mat srcImage, std::vector<double>& vecFeature);
	private:
		bool   readData(std::string fileName, std::vector<double> vecFeature, int label);
		double square(cv::Point a, cv::Point b, cv::Point c);
		double Solve(int l, int r, int m);
		void   gen_sobel_features(HObject ho_Image, HTuple hv_Features, HTuple *hv_FeaturesExtended);
		void   gen_features(HObject ho_Image, HTuple *hv_FeatureVector);
		bool   Mat2HImage(cv::Mat& InputImage, HObject& HSrcImage);
		bool   abstractGrayFeature(cv::Mat srcImage, std::vector<double>& vecFeature);
		bool   bin(cv::Mat srcImage);
		bool   getMaxtCountour();
		int    maxInnerCircle();
		bool   getLengthWidth(double& minRectLength, double& minRectWidth);
		double AreaOfConvexHull();
		double BoundaryPerimeter();
		double Area();
		double shapeFactor();
		double appearanceRatio();
		double enclosureArea();
		double shapeAngle();
		double rectangles();
		double roundness();
		double saturation();
		double eccentricity();
		double sphericity();
		double circularity();

		int getMax(int a, int b);
	private:
		cv::Mat                binImage;
		std::vector<cv::Point> hullPoint;
		std::vector<cv::Point> maxCountour;
		cv::Point2f            Pt[4];
		int                    flag;
		float                  radius;
		cv::Point2f            center;

		struct
		{
			bool operator()(cv::Point a, cv::Point b) const
			{
				return a.x < b.x;

			}

		}cmp;
	};
}

#endif // !FEATURE_H_