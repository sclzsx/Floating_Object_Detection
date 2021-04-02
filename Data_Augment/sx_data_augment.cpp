#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//平移
void translation(const cv::Mat src, cv::Mat& dst1, cv::Mat& dst2, cv::Mat& dst3, cv::Mat& dst4, int dx, int dy)
{
	cout << dx << endl;
	cout << dy << endl;

	int newwidth = src.cols - dx;
	int newcols = src.rows - dy;
	Rect rect1(0, 0, newwidth, newcols);
	Rect rect2(dx, 0, newwidth, newcols);
	Rect rect3(0, dy, newwidth, newcols);
	Rect rect4(dx, dy, newwidth, newcols);

	Mat tmp1 = src.clone();
	Mat tmp2 = src.clone();
	Mat tmp3 = src.clone();
	Mat tmp4 = src.clone();

	dst1 = tmp1(rect1);
	dst2 = tmp2(rect2);
	dst3 = tmp3(rect3);
	dst4 = tmp4(rect4);

	//rectangle(src, rect1, Scalar(0, 255, 0));
	//rectangle(src, rect2, Scalar(255, 255, 0));
	//rectangle(src, rect3, Scalar(0, 255, 255));
	//rectangle(src, rect4, Scalar(0, 0, 255));

	resize(dst1, dst1, src.size());
	resize(dst2, dst2, src.size());
	resize(dst3, dst3, src.size());
	resize(dst4, dst4, src.size());
}

bool AdaptGammaEnhance(cv::Mat Src, cv::Mat &Dst)
{
	cv::Mat Img = Src.clone();
	cv::cvtColor(Img, Img, COLOR_BGR2HSV);
	cv::Mat imageHSV[3];
	cv::split(Img, imageHSV);
	cv::Mat ImgV = imageHSV[2].clone();
	ImgV.convertTo(ImgV, CV_32FC1);
	cv::Mat imageVNomalize = cv::Mat::zeros(Src.size(), CV_32FC1);
	imageVNomalize = ImgV / 255.0;

	cv::Scalar     mean;
	cv::Scalar     dev;
	cv::meanStdDev(imageVNomalize, mean, dev);
	float       fMean = mean.val[0];
	float       fDev = dev.val[0];
	//std::cout << fMean << ",\t" << fDev << std::endl;

	float Gamma;
	float c;
	float k;
	int H;
	cv::Mat ImgOut = cv::Mat::zeros(Src.size(), CV_32FC1);
	H = (0.5 - fMean > 0) ? 1 : 0;

	if (fDev <= 0.0833) //低对比度图像 4fDev <= 1/12
	{
		Gamma = -(log(fDev) / log(2));
	}
	else
	{
		Gamma = exp((1 - (fMean + fDev)) / 2);
	}
	//std::cout << ",\t" << Gamma << std::endl;
	for (int i = 0; i < ImgOut.rows; i++)
	{
		for (int j = 0; j < ImgOut.cols; j++)
		{
			float IinGamma = pow(imageVNomalize.at<float>(i, j), Gamma);
			k = IinGamma + (1 - IinGamma)*pow(fMean, Gamma);
			c = 1.0 / (1 + H * (k - 1));
			ImgOut.at<float>(i, j) = c * pow(imageVNomalize.at<float>(i, j), Gamma)*255.0;
		}
	}
	//cv::normalize(ImgOut, ImgOut, 0, 255, CV_MINMAX);
	//转换成8bit图像显示  
	cv::convertScaleAbs(ImgOut, ImgOut);
	//ImgOut.convertTo(ImgOut, CV_8UC1);

	imageHSV[2] = ImgOut.clone();
	cv::merge(imageHSV, 3, Dst);
	cv::cvtColor(Dst, Dst, COLOR_HSV2BGR);
	//imshow("AdaGamma", Dst);
	return true;
}
bool HistogramEqualization(cv::Mat Src, cv::Mat &equ)
{
	Mat imageRGB[3];
	split(Src, imageRGB);
	for (int i = 0; i < 3; i++)
	{
		equalizeHist(imageRGB[i], imageRGB[i]);
	}
	merge(imageRGB, 3, equ);
	//imshow("直方图均衡化图像增强效果", equ);
	return 0;
}
bool LaplacianEnhance(cv::Mat Src, cv::Mat &Lap)
{
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
	filter2D(Src, Lap, CV_8UC3, kernel);
	//imshow("拉普拉斯算子图像增强效果", Lap);
	return 0;
}
bool LogarithmicTransformation(cv::Mat Src, cv::Mat &imageLog)
{
	imageLog = Mat::zeros(Src.size(), CV_32FC3);
	for (int i = 0; i < Src.rows; i++)
	{
		for (int j = 0; j < Src.cols; j++)
		{
			imageLog.at<Vec3f>(i, j)[0] = log(1 + Src.at<Vec3b>(i, j)[0]);
			imageLog.at<Vec3f>(i, j)[1] = log(1 + Src.at<Vec3b>(i, j)[1]);
			imageLog.at<Vec3f>(i, j)[2] = log(1 + Src.at<Vec3b>(i, j)[2]);
		}
	}
	//归一化到0~255  
	normalize(imageLog, imageLog, 0, 255, NORM_MINMAX);
	//转换成8bit图像显示  
	convertScaleAbs(imageLog, imageLog);
	//imshow("log", imageLog);
	return 0;
}
bool GammaTransform(cv::Mat Src, cv::Mat &imageGamma)
{
	imageGamma = Mat::zeros(Src.size(), CV_32FC3);
	for (int i = 0; i < Src.rows; i++)
	{
		for (int j = 0; j < Src.cols; j++)
		{
			imageGamma.at<Vec3f>(i, j)[0] = (Src.at<Vec3b>(i, j)[0])*(Src.at<Vec3b>(i, j)[0])*(Src.at<Vec3b>(i, j)[0]);
			imageGamma.at<Vec3f>(i, j)[1] = (Src.at<Vec3b>(i, j)[1])*(Src.at<Vec3b>(i, j)[1])*(Src.at<Vec3b>(i, j)[1]);
			imageGamma.at<Vec3f>(i, j)[2] = (Src.at<Vec3b>(i, j)[2])*(Src.at<Vec3b>(i, j)[2])*(Src.at<Vec3b>(i, j)[2]);
		}
	}
	//归一化到0~255  
	normalize(imageGamma, imageGamma, 0, 255, NORM_MINMAX);
	//转换成8bit图像显示  
	convertScaleAbs(imageGamma, imageGamma);
	//imshow("伽马变换图像增强效果", imageGamma);
	return 0;
}

