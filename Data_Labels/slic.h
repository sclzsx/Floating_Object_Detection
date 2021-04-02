#ifndef SLIC_H
#define SLIC_H

#include <vector>
#include <string>
#include <algorithm>
#include <ctime>
#include <memory.h>
#include <windows.h>
#include <opencv2/opencv.hpp>

#if (defined WIN32 || defined _WIN32)
#  define SLIC_EXPORTS __declspec(dllexport)
#else
#  define SLIC_EXPORTS
#endif

static int       number;

typedef unsigned int UINT;
typedef unsigned char uchar;

enum imageType { RGB, GRAY };

namespace IPSG
{
	class SLIC_EXPORTS SLIC
	{
	public:
		SLIC();
		virtual ~SLIC();

		//===========================================================================
		///	Perform SLIC algorithm on the given image 
		/// with the given number of superpixels
		//===========================================================================
		void GenerateSuperpixels(
			cv::Mat& img,
			UINT numSuperpixels
		);
		
		void GenerateSuperpixels2(
			cv::Mat& img,
			UINT numSuperpixels
		);
		//显著性目标检测
		//===========================================================================
		///	Get label on each pixel which shows the number of superpixel it belongs to
		//===========================================================================
		int* GetLabel();

		//===========================================================================
		///	Get the result image with contours on the given color
		//===========================================================================
		cv::Mat GetImgWithContours(const int&					K,
			cv::Scalar color);

		///////////////以下为自定义函数及成员变量///////////////////////////////////////////////////////////

		//Get the samples of the same object in different scales
		bool DrawScalesRects(cv::Point PointXY);		//获得不同尺度的样本及矩形区域
		bool SuperpixelTest(cv::Mat Img, int numSuperpixel = 1200);				//超像素分割测试，输入为视频帧，将输出参数保存至成员变量中
		bool SuperpixelTest2(cv::Mat Img, int numSuperpixel = 1700);		    //超像素分割测试，输入为视频帧，将输出参数保存至成员变量中

		struct SOutSamples		//成员变量，保存输出参数
		{
			cv::Mat m_ImgSrc;
			std::vector<cv::Mat> m_vOutSamples;		//每帧图像中截取的样本
			std::vector<cv::Rect> m_vOutRects;		//每个样本对应的图像矩形区域
			std::vector<cv::Point> m_CenterPoints;	//每个样本区域中心点
			int SaliencyWidth;                      //截取的显著性区域的宽
			int SaliencyHeight;                     //截取的显著性区域的高
		}m_SOutSamples;

	private:

		//============================================================================
		// Superpixel segmentation for a given step size (superpixel size ~= step*step)
		//============================================================================
		void PerformSLICO_ForGivenStepSize(
			unsigned int*			ubuff,//Each 32 bit unsigned int contains ARGB pixel values.
			const int					width,
			const int					height,
			int*						klabels,
			int&						numlabels,
			const int&					STEP,
			const double&				m

		);
		//============================================================================
		// Superpixel segmentation for a given number of superpixels
		//============================================================================
		void PerformSLICO_ForGivenK(
			unsigned int*			ubuff,//Each 32 bit unsigned int contains ARGB pixel values.
			const int					width,
			const int					height,
			int*						klabels,
			int&						numlabels,
			const int&					K,
			const double&				m,
			cv::Scalar color
		);
		
		void PerformSLICO_ForGivenK2(
			unsigned int*			ubuff,//Each 32 bit unsigned int contains ARGB pixel values.
			const int					width,
			const int					height,
			int*						klabels,
			int&						numlabels,
			const int&					K,
			const double&				m,
			cv::Scalar color
		);

		void SaveImage(cv::Mat inputImage, std::string path); 

		void PerformSLICO_ForGivenK(
			unsigned char*		ubuff,
			const int					width,
			const int					height,
			int*						klabels,
			int&						numlabels,
			const int&					K,//required number of superpixels
			const double&				m);//weight given to spatial distance

										   //============================================================================
										   // Save superpixel labels in a text file in raster scan order
										   //============================================================================
		void SaveSuperpixelLabels(
			const int*					labels,
			const int&					width,
			const int&					height,
			const std::string&				filename,
			const std::string&				path);
		//============================================================================
		// Function to draw boundaries around superpixels of a given 'color'.
		// Can also be used to draw boundaries around supervoxels, i.e layer by layer.
		//============================================================================
		void DrawContoursAroundSegments(
			std::vector<double>&				kseedsl,
			std::vector<double>&				kseedsa,
			std::vector<double>&				kseedsb,
			std::vector<double>&				kseedsx,
			std::vector<double>&				kseedsy,
			unsigned int*				ubuff,
			const int*					labels,
			const int&					width,
			const int&					height,
			const cv::Scalar&			color);

		void DrawContoursAroundSegments(
			std::vector<double>&				kseedsl,
			std::vector<double>&				kseedsa,
			std::vector<double>&				kseedsb,
			std::vector<double>&				kseedsx,
			std::vector<double>&				kseedsy,
			unsigned char*			ubuff,
			const int*				labels,
			const int&				width,
			const int&				height,
			const cv::Scalar&		color);

		void DrawContoursAroundSegmentsTwoColors(
			unsigned int*				ubuff,
			const int*					labels,
			const int&					width,
			const int&					height);



	private:

		//============================================================================
		// Magic SLIC. No need to set M (compactness factor) and S (step size).
		// SLICO (SLIC Zero) varies only M dynamicaly, not S.
		//============================================================================


		void PerformSuperpixelSegmentation_VariableSandM(
			std::vector<double>&				kseedsl,
			std::vector<double>&				kseedsa,
			std::vector<double>&				kseedsb,
			std::vector<double>&				kseedsx,
			std::vector<double>&				kseedsy,
			int*						klabels,
			const int&					STEP,
			const int&					NUMITR
		);
		//============================================================================
		// Pick seeds for superpixels when step size of superpixels is given.
		//============================================================================
		void GetLABXYSeeds_ForGivenStepSize(
			std::vector<double>&				kseedsl,
			std::vector<double>&				kseedsa,
			std::vector<double>&				kseedsb,
			std::vector<double>&				kseedsx,
			std::vector<double>&				kseedsy,
			const int&					STEP,
			const bool&					perturbseeds,
			const std::vector<double>&		edgemag);
		//============================================================================
		// Pick seeds for superpixels when number of superpixels is input.
		//============================================================================
		void GetLABXYSeeds_ForGivenK(
			std::vector<double>&				kseedsl,
			std::vector<double>&				kseedsa,
			std::vector<double>&				kseedsb,
			std::vector<double>&				kseedsx,
			std::vector<double>&				kseedsy,
			const int&					STEP,
			const bool&					perturbseeds,
			const std::vector<double>&		edges);

		//============================================================================
		// Move the seeds to low gradient positions to avoid putting seeds at region boundaries.
		//============================================================================
		void PerturbSeeds(
			std::vector<double>&				kseedsl,
			std::vector<double>&				kseedsa,
			std::vector<double>&				kseedsb,
			std::vector<double>&				kseedsx,
			std::vector<double>&				kseedsy,
			const std::vector<double>&		edges);
		//============================================================================
		// Detect color edges, to help PerturbSeeds()
		//============================================================================
		void DetectLabEdges(
			const double*				lvec,
			const double*				avec,
			const double*				bvec,
			const int&					width,
			const int&					height,
			std::vector<double>&				edges);
		//============================================================================
		// xRGB to XYZ conversion; helper for RGB2LAB()
		//============================================================================
		void RGB2XYZ(
			const int&					sR,
			const int&					sG,
			const int&					sB,
			double&						X,
			double&						Y,
			double&						Z);
		//============================================================================
		// sRGB to CIELAB conversion
		//============================================================================
		void RGB2LAB(
			const int&					sR,
			const int&					sG,
			const int&					sB,
			double&						lval,
			double&						aval,
			double&						bval);
		//============================================================================
		// sRGB to CIELAB conversion for 2-D images
		//============================================================================
		void DoRGBtoLABConversion(
			unsigned int*&		ubuff,
			double*&					lvec,
			double*&					avec,
			double*&					bvec);
		//============================================================================
		// sRGB to CIELAB conversion for 3-D volumes
		//============================================================================
		void DoRGBtoLABConversion(
			unsigned int**&		ubuff,
			double**&					lvec,
			double**&					avec,
			double**&					bvec);

		//============================================================================
		// Post-processing of SLIC segmentation, to avoid stray labels.
		//============================================================================
		void EnforceLabelConnectivity(
			std::vector<double>& nlabelcount,
			const int*					labels,
			const int&					width,
			const int&					height,
			int*						nlabels,//input labels that need to be corrected to remove stray labels
			int&						numlabels,//the number of labels changes in the end if segments are removed
			const int&					K); //the number of superpixels desired by the user

		void Mat2Buffer(const cv::Mat& img, UINT*& buffer);

		void Mat2Buffer(const cv::Mat& img, uchar*& buffer);

		/*bool Mergearea(
			std::vector<double>&        kseedsl,
			std::vector<double>&        kseedsa,
			std::vector<double>&        kseedsb,
			std::vector<double>&        kseedsx,
			std::vector<double>&        kseedsy,
			std::vector<double>&        nlabel,
			unsigned int*			ubuff,
			int*						nlabels,
			const int&					width,
			const int&					height,
			cv::Scalar color
		);*/

		void Mergearea2(
			std::vector<double>&        kseedsl,
			std::vector<double>&        kseedsa,
			std::vector<double>&        kseedsb,
			std::vector<double>&        kseedsx,
			std::vector<double>&        kseedsy,
			std::vector<double>&        nlabel,
			unsigned int*			ubuff,
			int*						nlabels,
			const int&					width,
			const int&					height,
			cv::Scalar color
			);



	private:
		int										m_width;
		int										m_height;
		int										m_depth;

		int                                     m_nRectX ;
		int                                     m_nRectY ;

		double*									m_lvec;
		double*									m_avec;
		double*									m_bvec;

		double**								m_lvecvec;
		double**								m_avecvec;
		double**								m_bvecvec;

		UINT*									bufferRGB; // buffer for if RGB image
		uchar*									bufferGray; // buffer if gray image

		int*									label; // label record which superpixel a pixel belongs to
		imageType								type;
	};
}

#endif