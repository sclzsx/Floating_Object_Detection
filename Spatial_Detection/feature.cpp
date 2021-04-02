#include "feature.h"

#include <opencv2/imgproc/imgproc_c.h>

void IPSG::Feature::gen_features(HObject ho_Image, HTuple *hv_FeatureVector)
{
	HObject  ho_Zoomed1;

	(*hv_FeatureVector) = HTuple();
	//Compute features.
	gen_sobel_features(ho_Image, (*hv_FeatureVector), &(*hv_FeatureVector));
	//Downscale the image (image pyramid) and compute features.
	ZoomImageFactor(ho_Image, &ho_Zoomed1, 0.5, 0.5, "constant");
	gen_sobel_features(ho_Zoomed1, (*hv_FeatureVector), &(*hv_FeatureVector));
	//Uncomment lines to use further pyramid levels:
	//zoom_image_factor (Zoomed1, Zoomed2, 0.5, 0.5, 'constant')
	//gen_sobel_features (Zoomed2, FeatureVector, FeatureVector)
	//zoom_image_factor (Zoomed2, Zoomed3, 0.5, 0.5, 'constant')
	//gen_sobel_features (Zoomed3, FeatureVector, FeatureVector)
	//zoom_image_factor (Zoomed3, Zoomed4, 0.5, 0.5, 'constant')
	//gen_sobel_features (Zoomed4, FeatureVector, FeatureVector)
	(*hv_FeatureVector) = (*hv_FeatureVector).TupleReal();
	return;
}

void IPSG::Feature::gen_sobel_features(HObject ho_Image, HTuple hv_Features, HTuple *hv_FeaturesExtended)
{
	// Local iconic variables
	HObject  ho_EdgeAmplitude;

	// Local control variables
	HTuple  hv_Energy, hv_Correlation, hv_Homogeneity;
	HTuple  hv_Contrast, hv_AbsoluteHistoEdgeAmplitude, hv_Entropy;
	HTuple  hv_Anisotropy, hv_AbsoluteHistoImage;

	//Coocurrence matrix for 90 deg:
	CoocFeatureImage(ho_Image, ho_Image, 6, 90, &hv_Energy, &hv_Correlation, &hv_Homogeneity, &hv_Contrast);
	//Absolute histogram of edge amplitudes:
	SobelAmp(ho_Image, &ho_EdgeAmplitude, "sum_abs", 3);
	GrayHistoAbs(ho_EdgeAmplitude, ho_EdgeAmplitude, 8, &hv_AbsoluteHistoEdgeAmplitude);

	//You could of course compute more features:
	//Entropy and anisotropy:
	EntropyGray(ho_Image, ho_Image, &hv_Entropy, &hv_Anisotropy);
	//Absolute histogram of gray values:
	GrayHistoAbs(ho_Image, ho_Image, 8, &hv_AbsoluteHistoImage);
	//Add features to feature vector:
	(*hv_FeaturesExtended).Clear();
	(*hv_FeaturesExtended).Append(hv_Features);
	(*hv_FeaturesExtended).Append(hv_Energy);
	(*hv_FeaturesExtended).Append(hv_Correlation);
	(*hv_FeaturesExtended).Append(hv_Homogeneity);
	(*hv_FeaturesExtended).Append(hv_Contrast);
	(*hv_FeaturesExtended) = (*hv_FeaturesExtended).TupleConcat(hv_AbsoluteHistoEdgeAmplitude);
	//Activate the following lines to add the additional features you activated:
	(*hv_FeaturesExtended) = ((*hv_FeaturesExtended).TupleConcat(hv_Entropy)).TupleConcat(hv_Anisotropy);
	(*hv_FeaturesExtended) = (*hv_FeaturesExtended).TupleConcat(hv_AbsoluteHistoImage);
	return;
}



bool IPSG::Feature::mHuMoment(std::string inputImageFile, std::vector<double>& vecFeature)
{
	//IplImage* img = cvLoadImage(inputImageFile.c_str(), 0); 
	cv::Mat matimg = cv::imread(inputImageFile, 0);
	IplImage* img = &IplImage(matimg); 
	//IplImage *input = cvCloneImage(matimg);
	
	int bmpWidth = img->width;
	int bmpHeight = img->height;
	int bmpStep = img->widthStep;
	int bmpChannels = img->nChannels;
	uchar*pBmpBuf = (uchar*)img->imageData;

	double m00 = 0, m11 = 0, m20 = 0, m02 = 0, m30 = 0, m03 = 0, m12 = 0, m21 = 0; //中心矩  
	double x0 = 0, y0 = 0; //计算中心距时所使用的临时变量（x-x'）  
	double u20 = 0, u02 = 0, u11 = 0, u30 = 0, u03 = 0, u12 = 0, u21 = 0;//规范化后的中心矩  
																		 //double M[7]; //HU不变矩  
	double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;//临时变量，  
												  //double Center_x=0,Center_y=0;//重心  
	int Center_x = 0, Center_y = 0;//重心  
	int i, j; //循环变量  

			  // 获得图像的区域重心  
	double s10 = 0, s01 = 0, s00 = 0; //0阶矩和1阶矩 //注：二值图像的0阶矩表示面积  
	for (j = 0; j < bmpHeight; j++)//y  
	{
		for (i = 0; i < bmpWidth; i++)//x  
		{
			s10 += i*pBmpBuf[j*bmpStep + i];
			s01 += j*pBmpBuf[j*bmpStep + i];
			s00 += pBmpBuf[j*bmpStep + i];
		}
	}
	Center_x = (int)(s10 / s00 + 0.5);
	Center_y = (int)(s01 / s00 + 0.5);

	// 计算二阶、三阶矩  
	m00 = s00;
	for (j = 0; j < bmpHeight; j++)
	{
		for (i = 0; i < bmpWidth; i++)//x  
		{
			x0 = (i - Center_x);
			y0 = (j - Center_y);
			m11 += x0*y0*pBmpBuf[j*bmpStep + i];
			m20 += x0*x0*pBmpBuf[j*bmpStep + i];
			m02 += y0*y0*pBmpBuf[j*bmpStep + i];
			m03 += y0*y0*y0*pBmpBuf[j*bmpStep + i];
			m30 += x0*x0*x0*pBmpBuf[j*bmpStep + i];
			m12 += x0*y0*y0*pBmpBuf[j*bmpStep + i];
			m21 += x0*x0*y0*pBmpBuf[j*bmpStep + i];
		}
	}

	// 计算规范化后的中心矩  
	u20 = m20 / pow(m00, 2);
	u02 = m02 / pow(m00, 2);
	u11 = m11 / pow(m00, 2);
	u30 = m30 / pow(m00, 2.5);
	u03 = m03 / pow(m00, 2.5);
	u12 = m12 / pow(m00, 2.5);
	u21 = m21 / pow(m00, 2.5);

	// 计算中间变量。  
	t1 = (u20 - u02);
	t2 = (u30 - 3 * u12);
	t3 = (3 * u21 - u03);
	t4 = (u30 + u12);
	t5 = (u21 + u03);
	double M[7] = { 0 };
	// 计算不变矩  
	M[0] = u20 + u02;
	M[1] = t1*t1 + 4 * u11*u11;
	M[2] = t2*t2 + t3*t3;
	M[3] = t4*t4 + t5*t5;
	M[4] = t2*t4*(t4*t4 - 3 * t5*t5) + t3*t5*(3 * t4*t4 - t5*t5);
	M[5] = t1*(t4*t4 - t5*t5) + 4 * u11*t4*t5;
	M[6] = t3*t4*(t4*t4 - 3 * t5*t5) - t2*t5*(3 * t4*t4 - t5*t5);

	vecFeature.push_back(M[0]);
	vecFeature.push_back(M[1]);
	vecFeature.push_back(M[2]);
	vecFeature.push_back(M[3]);
	vecFeature.push_back(M[4]);
	vecFeature.push_back(M[5]);
	vecFeature.push_back(M[6]);

	return true;
}

bool IPSG::Feature::Mat2HImage(cv::Mat& InputImage, HObject& HSrcImage)
{
	if (InputImage.empty())
	{
		return false;
	}

	if (InputImage.channels() == 1)
	{
		int height = InputImage.rows;
		int width = InputImage.cols;
		uchar *dataGray = new uchar[width*height];
		for (int i = 0; i < height; i++)
		{
			memcpy(dataGray + width*i, InputImage.data + InputImage.step*i, width);
		}
		GenImage1(&HSrcImage, "byte", InputImage.cols, InputImage.rows, (Hlong)(dataGray));
		delete[] dataGray;
	}
	if (InputImage.channels() == 3)
	{
		int height = InputImage.rows;
		int width = InputImage.cols;
		cv::Mat  ImageRed, ImageGreen, ImageBlue;
		ImageRed = cv::Mat(height, width, CV_8UC1);
		ImageGreen = cv::Mat(height, width, CV_8UC1);
		ImageBlue = cv::Mat(height, width, CV_8UC1);
		std::vector<cv::Mat> ImageChannels;
		split(InputImage, ImageChannels);

		ImageBlue = ImageChannels.at(0);
		ImageGreen = ImageChannels.at(1);
		ImageRed = ImageChannels.at(2);

		uchar*  dataRed = new uchar[InputImage.cols*InputImage.rows];
		uchar*  dataGreen = new uchar[InputImage.cols*InputImage.rows];
		uchar*  dataBlue = new uchar[InputImage.cols*InputImage.rows];
		for (int i = 0; i < height; i++)
		{
			memcpy(dataRed + width*i, ImageRed.data + ImageRed.step*i, width);
			memcpy(dataGreen + width*i, ImageGreen.data + ImageGreen.step*i, width);
			memcpy(dataBlue + width*i, ImageBlue.data + ImageBlue.step*i, width);
		}
		GenImage3(&HSrcImage, "byte", InputImage.cols, InputImage.rows, (Hlong)(dataRed), (Hlong)(dataGreen), (Hlong)(dataBlue));
		delete[]  dataRed;
		delete[]  dataGreen;
		delete[]  dataBlue;
	}
	return true;
}

bool IPSG::Feature::abstractFeature(cv::Mat srcImage, std::vector<double>& vecFeature)
{
	cv::Mat image = srcImage.clone();
	abstractGrayFeature(image, vecFeature);
	/*bin(image);
	double dAreaOfConvexHull = AreaOfConvexHull();
	double dBoundaryPerimeter = BoundaryPerimeter();
	double dArea = Area();
	double dShapeFactor = shapeFactor();
	double dAppearanceRatio = appearanceRatio();
	double dEnclosureArea = enclosureArea();
	double dShapeAngle = shapeAngle();
	double dRectangles = rectangles();
	double dRoundness = roundness();
	double dSaturation = saturation();
	double dEccentricity = eccentricity();
	double dSphericity = sphericity();
	double dCircularity = circularity();

	vecFeature.push_back(dAreaOfConvexHull);
	vecFeature.push_back(dAppearanceRatio);
	vecFeature.push_back(dBoundaryPerimeter);
	vecFeature.push_back(dArea);
	vecFeature.push_back(dShapeFactor);
	vecFeature.push_back(dAppearanceRatio);
	vecFeature.push_back(dEnclosureArea);
	vecFeature.push_back(dShapeAngle);
	vecFeature.push_back(dRoundness);
	vecFeature.push_back(dSaturation);
	vecFeature.push_back(dEccentricity);
	vecFeature.push_back(dSphericity);
	vecFeature.push_back(dCircularity);*/

	return true;
}

bool IPSG::Feature::abstractGrayFeature(cv::Mat srcImage, std::vector<double>& vecFeature)
{
	cv::Mat image = srcImage.clone();
	HTuple hvWidth, hvHeight;
	HTuple hvFeatureVector;
	HObject hoImage, hoImageResize;
	Mat2HImage(image, hoImage);
	GetImageSize(hoImage, &hvWidth, &hvHeight);
	Emphasize(hoImage, &hoImageResize, hvWidth, hvHeight, 1.0);
	gen_features(hoImage, &hvFeatureVector);

	for (int i = 0; i < hvFeatureVector.Length(); i++)
	{
		vecFeature.push_back(hvFeatureVector[i].D());
	}

	return true;
}

double IPSG::Feature::square(cv::Point a, cv::Point b, cv::Point c)
{
	return (a.x*b.y + b.x*c.y + c.x*a.y - a.x*c.y - b.x*a.y - c.x*b.y)*1.0 / 2;
}

double IPSG::Feature::Solve(int l, int r, int m)
{
	int mi = -1;
	double mdis = 0;
	for (int i = l + 1; i < r; i++)
	{
		double dis = square(hullPoint[l], hullPoint[r], hullPoint[i])*m;  //找到面积最大的点 同一个底，面积越大 高越高 距离越远
		if (dis > mdis)
		{
			mdis = dis;
			mi = i;
		}
	}
	if (mi == -1)
		return 0;
	return Solve(l, mi, m) + Solve(mi, r, m) + mdis;
}

bool IPSG::Feature::bin(cv::Mat srcImage)
{
	cv::Mat image = srcImage.clone();
	cv::Mat grayImage(image.rows, image.cols, CV_8UC1);
	binImage = cv::Mat(image.rows, image.cols, CV_8UC1);
	if (image.channels() == 1)
	{
		grayImage = image;
	}
	else if (image.channels() != 1 )
	{
		cv::cvtColor(image, grayImage, 6);
	}
	cv::threshold(grayImage, grayImage, 100, 255, CV_THRESH_OTSU);
	cv::medianBlur(grayImage, grayImage, 5);
	cv::morphologyEx(grayImage, binImage, CV_MOP_CLOSE, cv::getStructuringElement(0, cv::Size(3, 3)));

	return true;
}

bool IPSG::Feature::getMaxtCountour()
{
	cv::Mat draw = cv::Mat::zeros(binImage.size(), CV_8UC3);
	std::vector<std::vector<cv::Point>> contours;
	int imax = 0;
	int imaxContour = -1;
	cv::findContours(binImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours.size(); i++) {
		double itmp = cv::contourArea(contours[i]);//这里采用的是轮廓大小
		if (imaxContour < itmp) {
			imax = i;
			imaxContour = itmp;
		}
	}
	maxCountour = contours[imax];
	return true;
}

int IPSG::Feature::maxInnerCircle()
{
	cv::Point centerInnerCircle;
	int dist = 0;
	int maxInnerCircleRadius = 0;
	for (int i = 0; i < binImage.cols; i++)
	{
		for (int j = 0; j < binImage.rows; j++)
		{
			dist = cv::pointPolygonTest(maxCountour, cv::Point(i, j), true);
			if (dist > maxInnerCircleRadius)
			{
				maxInnerCircleRadius = dist;
				centerInnerCircle = cv::Point(i, j);
			}
		}
	}

	return maxInnerCircleRadius;
}

bool IPSG::Feature::getLengthWidth(double& minRectLength, double& minRectWidth)
{
	cv::RotatedRect box;
	box = cv::minAreaRect(cv::Mat(maxCountour));
	box.points(Pt);
	minRectLength = sqrt(abs(Pt[0].y - Pt[1].y)*abs(Pt[0].y - Pt[1].y) + abs(Pt[0].x - Pt[1].x)*abs(Pt[0].x - Pt[1].x));
	minRectWidth = sqrt(abs(Pt[1].y - Pt[2].y)*abs(Pt[1].y - Pt[2].y) + abs(Pt[1].x - Pt[2].x)*abs(Pt[1].x - Pt[2].x));
	if (minRectWidth > minRectLength)
	{
		flag = 1;
		double tempt = minRectLength;
		minRectLength = minRectWidth;
		minRectWidth = tempt;
	}

	return true;
}

double IPSG::Feature::AreaOfConvexHull()
{
	getMaxtCountour();
	cv::convexHull(maxCountour, hullPoint, false);
	std::sort(hullPoint.begin(), hullPoint.end(), cmp);

	return Solve(0, hullPoint.size() - 1, 1) + Solve(0, hullPoint.size() - 1, -1);
}

double IPSG::Feature::BoundaryPerimeter()
{
	return cv::arcLength(maxCountour, true);
}

double IPSG::Feature::Area()
{
	return cv::contourArea(maxCountour);
}

double IPSG::Feature::shapeFactor()
{
	return BoundaryPerimeter() / Area();
}

double IPSG::Feature::appearanceRatio()
{
	double minRectLength = 0.0;
	double minRectWidth = 0.0;
	getLengthWidth(minRectLength, minRectWidth);
	
	return minRectLength / minRectWidth;
}

double IPSG::Feature::enclosureArea()
{
	double minRectLength = 0.0;
	double minRectWidth = 0.0;
	getLengthWidth(minRectLength, minRectWidth);

	return minRectLength * minRectWidth;
}

double IPSG::Feature::shapeAngle()
{
	double d_shapeAngle = 0.0;
	if (flag == 0) {
		double Y = Pt[2].y - Pt[1].y;
		double X = Pt[2].x - Pt[1].x;
		d_shapeAngle = atan(Y / X);
	}
	else if (flag == 1) {
		double Y = Pt[1].y - Pt[0].y;
		double X = Pt[1].x - Pt[0].x;
		d_shapeAngle = atan(Y / X);
	}

	return d_shapeAngle;
}

double IPSG::Feature::rectangles()
{
	return Area() / enclosureArea();
}

double IPSG::Feature::saturation()
{
	return Area() / AreaOfConvexHull();
}

double IPSG::Feature::roundness()
{
	cv::minEnclosingCircle(maxCountour, center, radius);
	
	return Area() / (CV_PI * radius * radius);
}

double IPSG::Feature::eccentricity()
{
	double maxDistance = 0.0;
	for (int i = 0; i < maxCountour.size(); i++)
	{
		for (int j = i + 1; j < maxCountour.size(); j++)
		{
			double currentDistance = sqrt(abs(maxCountour.at(i).y - maxCountour.at(j).y)*abs(maxCountour.at(i).y - maxCountour.at(j).y)
				+ abs(maxCountour.at(i).x - maxCountour.at(j).x)*abs(maxCountour.at(i).x - maxCountour.at(j).x));
			if (currentDistance > maxDistance)
				maxDistance = currentDistance;
		}
	}
	double minDistance, tmpDistance;
	int m, n;
	for (m = 0; m < maxCountour.size(); m++)
	{
		for (n = m + 1; n < maxCountour.size(); n++)
		{
			tmpDistance = sqrt(abs(maxCountour.at(m).y - maxCountour.at(n).y)*abs(maxCountour.at(m).y - maxCountour.at(n).y)
				+ abs(maxCountour.at(m).x - maxCountour.at(n).x)*abs(maxCountour.at(m).x - maxCountour.at(n).x));
			if (tmpDistance == maxDistance)
				break;
		}
		if (tmpDistance == maxDistance)
			break;
	}

	int a, b;
	if ((maxCountour.at(m).x - maxCountour.at(n).x) != 0)
	{
		double k1 = (maxCountour.at(m).y - maxCountour.at(n).y) / (maxCountour.at(m).x - maxCountour.at(n).x);


		std::vector<double> minDistances;
		for (a = 0; a < maxCountour.size(); a++)
		{
			for (b = a + 1; b < maxCountour.size(); b++)
			{
				if ((maxCountour.at(a).x - maxCountour.at(b).x) != 0)
				{
					double k2 = (maxCountour.at(a).y - maxCountour.at(b).y) / (maxCountour.at(a).x - maxCountour.at(b).x);
					if ((abs(k1*k2) - 1) < 0.000001)
					{
						double currentDistance = sqrt(abs(maxCountour.at(a).y - maxCountour.at(b).y)*abs(maxCountour.at(a).y - maxCountour.at(b).y)
							+ abs(maxCountour.at(a).x - maxCountour.at(b).x)*abs(maxCountour.at(a).x - maxCountour.at(b).x));
						minDistances.push_back(currentDistance);

					}
				}
			}
		}
		sort(minDistances.begin(), minDistances.end());
		minDistance = minDistances.at(0);
	}

	return maxDistance / minDistance;
}

double IPSG::Feature::sphericity()
{
	return maxInnerCircle() / radius;
}

double IPSG::Feature::circularity()
{
	std::vector<double> conCirDist;
	for (int i = 0; i < maxCountour.size(); i++)
	{
		double tmp = sqrt(abs(maxCountour.at(i).y - center.y)*abs(maxCountour.at(i).y - center.y)
			+ abs(maxCountour.at(i).x - center.x)*abs(maxCountour.at(i).x - center.x));
		conCirDist.push_back(tmp);
	}
	double sum = 0, avg = 0;
	for (int i = 0; i < conCirDist.size(); i++)
	{
		sum += conCirDist.at(i);
	}
	avg = sum / conCirDist.size();

	double sum2 = 0, avg2 = 0;
	for (int i = 0; i < conCirDist.size(); i++)
	{
		double tmp = (avg - conCirDist.at(i))*(avg - conCirDist.at(i));
		sum2 += tmp;
	}
	avg2 = sum2 / conCirDist.size();

	return avg / avg2;
}

void IPSG::Feature::getFile(std::string path, std::vector<std::string>& files)
{
	long hFile = 0;
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFile(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

bool IPSG::Feature::readData(std::string fileName, std::vector<double> vecFeature, int label)
{
	std::ofstream writeData;
	writeData.open(fileName, std::ios::out | std::ios::in | std::ios::app);
	std::string strFeature;
	if (label == 1)
	{
		strFeature = std::to_string(1).append(" ");
		for (int i = 0; i < vecFeature.size(); i++)
		{
			strFeature += std::to_string(vecFeature.at(i)).append(" ");
		}
	}
	else if (label == -1)
	{
		strFeature = std::to_string(-1).append(" ");
		for (int i = 0; i < vecFeature.size(); i++)
		{
			strFeature += std::to_string(vecFeature.at(i)).append(" ");
		}
	}

	writeData << strFeature;
	writeData << "\n";

	writeData.close();
	return true;
}

int IPSG::Feature::getMax(int a, int b)
{
	return a > b ? a : b;
}

bool IPSG::Feature::abstractSampleFeature(std::string PsamplePath, int PsampleLabel, std::string NsamplePath, int NsampleLabel, std::string savePath)
{
	std::vector<std::string> Pfiles;
	std::vector<double> PvecFeature;
	std::vector<std::string> Nfiles;
	std::vector<double> NvecFeature;
	getFile(PsamplePath, Pfiles);
	getFile(NsamplePath, Nfiles);
	if (Pfiles.size()== 0 || Nfiles.size() == 0)
	{
		std::cout << "输入样本路径错误" << std::endl;
		return false;
	}
	int maxSampleNum = getMax(Pfiles.size(), Nfiles.size());
	for (int i = 0; i < maxSampleNum; i++)
	{
		if (Pfiles.size() > i)
		{
			mHuMoment(Pfiles.at(i), PvecFeature);
			abstractFeature(cv::imread(Pfiles.at(i)), PvecFeature);
			readData(savePath, PvecFeature, PsampleLabel);
			PvecFeature.clear();
		}
		if (Nfiles.size() > i)
		{
			mHuMoment(Nfiles.at(i), NvecFeature);
			abstractFeature(cv::imread(Nfiles.at(i)), NvecFeature);
			readData(savePath, NvecFeature, NsampleLabel);
			NvecFeature.clear();
		}

	}

	return true;
}