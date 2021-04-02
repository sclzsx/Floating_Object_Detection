#include "gen_features.h"

void get_hog_feature_vector(Mat input, vector<float> &output)
{
	if (input.channels() != 1)
		cvtColor(input, input, cv::COLOR_RGB2GRAY);
	if (input.size() != cv::Size(winsize, winsize))
		resize(input, input, cv::Size(winsize, winsize));
	cv::HOGDescriptor hog(cv::Size(winsize, winsize), cv::Size(winsize, winsize)
		, cv::Size(cellsize, cellsize), cv::Size(cellsize, cellsize), 9);
	hog.compute(input, output, cv::Size(winsize, winsize));
}

void get_fhog_feature_vector(Mat input, vector<float> &output)
{
	if (input.channels() != 1)
		cvtColor(input, input, cv::COLOR_RGB2GRAY);
	if (input.size() != cv::Size(winsize, winsize))
		resize(input, input, cv::Size(winsize, winsize));
	//��ȡfhog����ͼ
	int cell_size = cellsize;
	int size_patch[3];
	IplImage z_ipl = input;
	Mat FeaturesMap;
	CvLSVMFeatureMapCaskade *map;
	getFeatureMaps(&z_ipl, cell_size, &map);
	normalizeAndTruncate(map, 0.2f);
	PCAFeatureMaps(map);
	size_patch[0] = map->sizeY;
	size_patch[1] = map->sizeX;
	size_patch[2] = map->numFeatures;
	FeaturesMap = cv::Mat(cv::Size(map->numFeatures, map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
	FeaturesMap = FeaturesMap.t();
	freeFeatureMapObject(&map);
	//������������(sizeY*sizeX*numFeatures)*1
	Mat feature = FeaturesMap.reshape(1, 1);
	int dim = feature.cols;
	//cout << "***********" << dim << endl;
	output.clear();
	for (int i = 0; i < dim; i++)
	{
		float val = feature.at<float>(0, i);
		output.push_back(val);
	}
}

void feature_vec2mat(vector<float> input, Mat &output)
{
	for (int i = 0; i < input.size(); i++)
	{
		output.at<float>(0, i) = input[i];
	}
}

void ComputeLBPImage_Uniform(const Mat &srcImage, Mat &LBPImage)
{
	// ������飬�ڴ����
	CV_Assert(srcImage.depth() == CV_8U && srcImage.channels() == 1);
	LBPImage.create(srcImage.size(), srcImage.type());

	// ����LBPͼ
	// ����ԭͼ��߽磬���ڱ߽紦��
	Mat extendedImage;
	copyMakeBorder(srcImage, extendedImage, 1, 1, 1, 1, cv::BORDER_DEFAULT);

	// ����LBP �ȼ�ģʽ���ұ�
	//int table[256];
	//BuildUniformPatternTable(table);

	// LUT(256��ÿһ��ģʽ��Ӧ�ĵȼ�ģʽ)
	static const int table[256] = { 1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 15, 16, 17, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25,
		0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0
		, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42
		, 43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58 };

	// ����LBP
	int heightOfExtendedImage = extendedImage.rows;
	int widthOfExtendedImage = extendedImage.cols;
	int widthOfLBP = LBPImage.cols;
	uchar *rowOfExtendedImage = extendedImage.data + widthOfExtendedImage + 1;
	uchar *rowOfLBPImage = LBPImage.data;
	for (int y = 1; y <= heightOfExtendedImage - 2; ++y, rowOfExtendedImage += widthOfExtendedImage, rowOfLBPImage += widthOfLBP)
	{
		// ��
		uchar *colOfExtendedImage = rowOfExtendedImage;
		uchar *colOfLBPImage = rowOfLBPImage;
		for (int x = 1; x <= widthOfExtendedImage - 2; ++x, ++colOfExtendedImage, ++colOfLBPImage)
		{
			// ����LBPֵ
			int LBPValue = 0;
			if (colOfExtendedImage[0 - widthOfExtendedImage - 1] >= colOfExtendedImage[0])
				LBPValue += 128;
			if (colOfExtendedImage[0 - widthOfExtendedImage] >= colOfExtendedImage[0])
				LBPValue += 64;
			if (colOfExtendedImage[0 - widthOfExtendedImage + 1] >= colOfExtendedImage[0])
				LBPValue += 32;
			if (colOfExtendedImage[0 + 1] >= colOfExtendedImage[0])
				LBPValue += 16;
			if (colOfExtendedImage[0 + widthOfExtendedImage + 1] >= colOfExtendedImage[0])
				LBPValue += 8;
			if (colOfExtendedImage[0 + widthOfExtendedImage] >= colOfExtendedImage[0])
				LBPValue += 4;
			if (colOfExtendedImage[0 + widthOfExtendedImage - 1] >= colOfExtendedImage[0])
				LBPValue += 2;
			if (colOfExtendedImage[0 - 1] >= colOfExtendedImage[0])
				LBPValue += 1;

			colOfLBPImage[0] = table[LBPValue];

		} // x

	}// y
}

void ComputeLBPFeatureVector_Uniform(const Mat &srcImage, cv::Size cellSize, Mat &featureVector)
{
	// ������飬�ڴ����
	CV_Assert(srcImage.depth() == CV_8U && srcImage.channels() == 1);

	Mat LBPImage;
	ComputeLBPImage_Uniform(srcImage, LBPImage);

	// ����cell����
	int widthOfCell = cellSize.width;
	int heightOfCell = cellSize.height;
	int numberOfCell_X = srcImage.cols / widthOfCell;// X����cell�ĸ���
	int numberOfCell_Y = srcImage.rows / heightOfCell;

	// ���������ĸ���
	int numberOfDimension = 58 * numberOfCell_X*numberOfCell_Y;
	featureVector.create(1, numberOfDimension, CV_32FC1);
	featureVector.setTo(cv::Scalar(0));

	// ����LBP��������
	int stepOfCell = srcImage.cols;
	int index = -58;// cell�������������������������е���ʼλ��
	float *dataOfFeatureVector = (float *)featureVector.data;
	for (int y = 0; y <= numberOfCell_Y - 1; ++y)
	{
		for (int x = 0; x <= numberOfCell_X - 1; ++x)
		{
			index += 58;

			// ����ÿ��cell��LBPֱ��ͼ
			Mat cell = LBPImage(cv::Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			uchar *rowOfCell = cell.data;
			int sum = 0; // ÿ��cell�ĵȼ�ģʽ����
			for (int y_Cell = 0; y_Cell <= cell.rows - 1; ++y_Cell, rowOfCell += stepOfCell)
			{
				uchar *colOfCell = rowOfCell;
				for (int x_Cell = 0; x_Cell <= cell.cols - 1; ++x_Cell, ++colOfCell)
				{
					if (colOfCell[0] != 0)
					{
						// ��ֱ��ͼ��ת��Ϊ0~57��������colOfCell[0] - 1
						++dataOfFeatureVector[index + colOfCell[0] - 1];
						++sum;
					}
				}
			}

			// һ��Ҫ��һ��������������������ܴ�
			for (int i = 0; i <= 57; ++i)
				dataOfFeatureVector[index + i] /= sum;
		}
	}
}

void get_lbp_feature_vector(Mat input, vector<float> &output)
{
	if (input.channels() != 1)
		cvtColor(input, input, cv::COLOR_RGB2GRAY);
	if (input.size() != cv::Size(winsize, winsize))
		resize(input, input, cv::Size(winsize, winsize));

	Mat featureMap;
	ComputeLBPFeatureVector_Uniform(input, cv::Size(cellsize,cellsize), featureMap);

	Mat feature = featureMap.reshape(1, 1);
	int dim = feature.cols;
	//cout << "***********" << dim << endl;
	output.clear();
	for (int i = 0; i < dim; i++)
	{
		float val = feature.at<float>(0, i);
		output.push_back(val);
	}
}

void get_glcm_feature_vector(Mat input, vector<float> &output)
{
	if (input.channels() != 1)
		cvtColor(input, input, cv::COLOR_RGB2GRAY);
	if (input.size() != cv::Size(winsize, winsize))
		resize(input, input, cv::Size(winsize, winsize));

	std::vector<double> hal_tmp;
	hal_tmp.clear();
	IPSG::Feature halfea;
	halfea.abstractFeature(input, hal_tmp);
	
	int dim = hal_tmp.size();

	////double to float vector
	std::vector<float> hal_tmp2;
	for (int j = 0; j < dim; j++)
	{
		float tmp = abs((float)hal_tmp[j]);
		//cout << tmp << " ";
		hal_tmp2.push_back(tmp);
	}
	output = hal_tmp2;

	//////vector to mat and normalize
	//Mat feature_mat = Mat::zeros(1, dim, CV_32FC1);
	//feature_vec2mat(hal_tmp2, feature_mat);
	////cout << feature_mat << endl;
	//cv::Mat feamat4 = Mat::zeros(1, dim, CV_32FC1);
	//cv::normalize(feature_mat, feamat4, 0, 1, cv::NORM_MINMAX);
	////cout << feamat4 << endl;

	//////mat to vector
	//for (int j = 0; j < dim; j++)
	//{
	//	float tmp = feamat4.at<float>(0, j);
	//	//cout << tmp << " ";
	//	output.push_back(tmp);
	//}
	////cout << "finish3" << endl;
}

void get_fhog_glcm_feature_vector(Mat input, vector<float> &output)
{
	vector<float> vec1, vec2;
	get_fhog_feature_vector(input, vec1);
	get_glcm_feature_vector(input, vec2);
	output = vec1;
	output.insert(output.end(), vec2.begin(), vec2.end());
}