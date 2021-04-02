
#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <iostream>
#include <vector>
#include <fstream>

#include "HalconCpp.h"
#include "HDevThread.h"

#include <opencv2/opencv.hpp>

using namespace HalconCpp;

namespace IPSG
{
	//bool UpCameraHandleFinish = false;
	//bool DownCameraHandleFinish = false;
	class CClassifierMLP
	{
	public:
		CClassifierMLP();
		~CClassifierMLP();

		bool RecongitionClassifier(cv::Mat& InputImage, bool &Category);
		bool MatToHImage(cv::Mat& InputImage, HObject& HSrcImage);
		//bool IPSG::CClassifierMLP::RecongitionClassifier(cv::Mat InputImage);
		bool RecongitionClassifier(cv::Mat src_InputImage);
	private:
		void dev_update_off();
		void disp_continue_message(HTuple hv_WindowHandle, HTuple hv_Color, HTuple hv_Box);
		void disp_end_of_program_message(HTuple hv_WindowHandle, HTuple hv_Color, HTuple hv_Box);
		void disp_message(HTuple hv_WindowHandle, HTuple hv_String, HTuple hv_CoordSystem, HTuple hv_Row, HTuple hv_Column, HTuple hv_Color, HTuple hv_Box);
		void set_display_font(HTuple hv_WindowHandle, HTuple hv_Size, HTuple hv_Font, HTuple hv_Bold, HTuple hv_Slant);
		void gen_features(HObject ho_Image, HTuple *hv_FeatureVector);
		void gen_sobel_features(HObject ho_Image, HTuple hv_Features, HTuple *hv_FeaturesExtended);
	private:
		bool                        SaliencyFlag;            //分类器判别标志
	};
}
#endif