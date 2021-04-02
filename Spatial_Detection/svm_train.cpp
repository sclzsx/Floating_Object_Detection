#include "svm_train.h"
#include "fhog.h"
#include "feature.h"
#include <fstream>
#define GRAYFEATURE 0

int cellsize = 16;

int winsize = 80;
int stride = 10;

int descriptordim;

cv::Size dsize = cv::Size(winsize, winsize);

Mat trFeatureMat;
Mat trLabelMat;
Mat teFeatureMat;
Mat teLabelMat;

string trPosPath = "E:\\DataSets\\litter\\classify_V2\\train\\pos\\";
string trNegPath = "E:\\DataSets\\litter\\classify_V2\\train\\neg\\";
string trPosName = "E:\\DataSets\\litter\\classify_V2\\train\\pos.txt";
string trNegName = "E:\\DataSets\\litter\\classify_V2\\train\\neg.txt";
string tePosPath = "E:\\DataSets\\litter\\classify_V2\\test\\pos\\";
string teNegPath = "E:\\DataSets\\litter\\classify_V2\\test\\neg\\";
string tePosName = "E:\\DataSets\\litter\\classify_V2\\test\\pos.txt";
string teNegName = "E:\\DataSets\\litter\\classify_V2\\test\\neg.txt";

string detestTestName = "E:\\DataSets\\litter\\src_V2\\detect\\";

string classifier_name = "cell" + to_string(cellsize) + "_win" + to_string(winsize) + "_gray" + to_string(GRAYFEATURE)+"_classify_V2"+".xml";

void GetPath(string PosPath, string NegPath, string namepos, string nameneg, vector<string> &posSamplesPath, vector<string> &negSamplesPath)
{
	string buffer;
	std::ifstream fInPos(namepos);
	std::ifstream fInNeg(nameneg);
	while (fInPos)
	{
		if (getline(fInPos, buffer))
		{
			posSamplesPath.push_back(PosPath + buffer);
		}
	}
	fInPos.close();
	while (fInNeg)
	{
		if (getline(fInNeg, buffer))
		{
			negSamplesPath.push_back(NegPath + buffer);
		}
	}
	fInNeg.close();
}

void my_feature(Mat resized_gray, vector<float> &descriptors)
{
	//提取fhog特征图
	int cell_size = cellsize;
	int size_patch[3];
	IplImage z_ipl = resized_gray;
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
	//构建特征向量(sizeY*sizeX*numFeatures)*1
	Mat feature = FeaturesMap.reshape(1, 1);
	int dim = feature.cols;
	//cout << "***********" << dim << endl;
	descriptors.clear();
	for (int i = 0; i < dim; i++)
	{
		float val = feature.at<float>(0, i);
		descriptors.push_back(val);
	}

#if GRAYFEATURE
	std::vector<double> descriptor2;
	descriptor2.clear();
	IPSG::Feature halfea;
	halfea.abstractFeature(resized_gray, descriptor2);
	//int descriptordim2 = descriptor2.size();
	//cout << "###########" << descriptordim2 << endl;

	std::vector<double> descriptor3;
	for (int j = 0; j < descriptor2.size(); j++)
	{
		descriptor3.push_back((float)descriptor2.at(j));
	}
	descriptors.insert(descriptors.end(), descriptor3.begin(), descriptor3.end());
	//int descriptordim3 = descriptor3.size();
	//cout << "%%%%%%%%%%%" << descriptordim3 << endl;
#endif
}

void my_feature2(Mat resized_gray, Mat &feature)
{
	std::vector<float> tmp;
	my_feature(resized_gray, tmp);
	for (int i=0;i<tmp.size();i++)
	{
		feature.at<float>(0, i) = tmp[i];
	}
}

void GetFeatureAndLabel(string PosPath, string NegPath, string namepos, string nameneg, Mat &FeatureMat, Mat &LabelMat)
{
	vector<string> posSamples;
	vector<string> negSamples;
	GetPath(PosPath, NegPath, namepos, nameneg, posSamples, negSamples);
	int posSampleNum = posSamples.size();
	int negSampleNum = negSamples.size();

	Mat tmp1 = cv::imread(posSamples[0]);
	Mat tmp2 = Mat(dsize, CV_32S);
	resize(tmp1, tmp2, dsize);
	cvtColor(tmp2, tmp2, CV_BGR2GRAY);
	vector<float> tmp;
	tmp.clear();
	my_feature(tmp2, tmp);
	descriptordim = tmp.size();
	cout << "$$$$$$$$$$$$$$$$$$$$$ dim is: " << descriptordim << endl;
	//imshow("testImg", tmp2);
	//cv::waitKey();

	FeatureMat = Mat::zeros(posSampleNum + negSampleNum, descriptordim, CV_32FC1);
	LabelMat = Mat::zeros(posSampleNum + negSampleNum, 1, CV_32S);

	for (int i = 0; i < posSampleNum; i++)
	{
		Mat inputImg = cv::imread(posSamples[i]);
		std::cout << "processing " << i << "/" << posSampleNum << " " << posSamples[i] << endl;
		Mat trainImg = Mat(dsize, CV_32S);
		resize(inputImg, trainImg, dsize);
		cvtColor(trainImg, trainImg, CV_BGR2GRAY);

		vector<float> descriptor;
		descriptor.clear();
		my_feature(trainImg, descriptor);
		for (int j = 0; j < descriptordim; j++)
		{
			FeatureMat.at<float>(i, j) = descriptor[j];
		}
		LabelMat.at<int>(i, 0) = 1;
	}
	std::cout << "extract pos Feature done" << "dim is "<< descriptordim <<endl;

	for (int i = 0; i < negSampleNum; i++)
	{
		Mat inputImg = cv::imread(negSamples[i]);
		std::cout << "processing " << i << "/" << negSampleNum << " " << negSamples[i] << endl;
		Mat trainImg = Mat(dsize, CV_32S);
		resize(inputImg, trainImg, dsize);
		cvtColor(trainImg, trainImg, CV_BGR2GRAY);
		
		vector<float> descriptor;
		descriptor.clear();
		my_feature(trainImg, descriptor);

		for (int j = 0; j < descriptordim; j++)
		{
			FeatureMat.at<float>(posSampleNum + i, j) = descriptor[j];
		}
		LabelMat.at<int>(posSampleNum + i, 0) = -1;
	}
	std::cout << "extract neg Feature done" << "dim is " << descriptordim << endl;
}

void GetFeatureMat(vector<Mat> inputMats, Mat &FeatureMat)
{
	int num = inputMats.size();
	FeatureMat = Mat::zeros(num, descriptordim, CV_32FC1);
	for (int i = 0; i < num; i++)
	{
		Mat trainImg = Mat(dsize, CV_32S);
		resize(inputMats[i], trainImg, dsize);
		cvtColor(trainImg, trainImg, CV_BGR2GRAY);

		vector<float> descriptor;
		descriptor.clear();
		my_feature(trainImg, descriptor);
		//int descriptordim = descriptor.size();

		for (int j = 0; j < descriptordim; j++)
		{
			FeatureMat.at<float>(i, j) = descriptor[j];
		}
	}
}

void load_images(const string dirname, vector< Mat > &img_lst, bool gray)
{
	vector< string > files;
	cv::glob(dirname, files);
	for (size_t i = 0; i < files.size(); ++i)
	{
		Mat img = cv::imread(files[i]); // load the image
		if (img.empty())            // invalid image, skip it.
		{
			cout << files[i] << " is invalid!" << endl;
			continue;
		}
		if (gray)
		{
			cvtColor(img, img, CV_BGR2GRAY);
		}
		img_lst.push_back(img);
	}
}

void save_data(string dataname, string labelname, string inputpath1, string inputpath2)
{
	vector< Mat > outputmats1, outputmats2;
	load_images(inputpath1, outputmats1, 1);
	load_images(inputpath2, outputmats2, 1);
	int total_num = outputmats1.size() + outputmats2.size();

	vector<float> tmp;
	my_feature(outputmats1[0], tmp);
	descriptordim = tmp.size();
	cout << "$$$$$$$$$$$$$$$$$$$$$ dim is: " << descriptordim << endl;

	Mat FeatureMat = Mat::zeros(total_num, descriptordim, CV_32FC1);
	Mat LabelMat = Mat::zeros(total_num, 1, CV_32S);
	for (int i = 0; i < outputmats1.size(); i++)
	{
		vector<float> descriptor;
		my_feature(outputmats1[i], descriptor);
		for (int j = 0; j < descriptordim; j++)
		{
			FeatureMat.at<float>(i, j) = descriptor[j];
		}
		LabelMat.at<int>(i, 0) = 1;
	}
	for (int i = 0; i < outputmats2.size(); i++)
	{
		vector<float> descriptor;
		my_feature(outputmats2[i], descriptor);
		for (int j = 0; j < descriptordim; j++)
		{
			FeatureMat.at<float>(i + outputmats1.size(), j) = descriptor[j];
		}
		LabelMat.at<int>(i + outputmats1.size(), 0) = 2;
	}
	cout << FeatureMat.rows << " " << FeatureMat.cols << " " << total_num << endl;
	std::fstream out1(dataname, std::ios::ate | std::ios::out);
	std::fstream out2(labelname, std::ios::ate | std::ios::out);
	for (int i = 0; i < FeatureMat.rows; i++)
	{
		for (int j = 0; j < descriptordim; j++)
			out1 << FeatureMat.at<float>(i, j) << ",  ";
		out1 << endl;
	}
	out1.close();
	for (int i=0;i<LabelMat.rows;i++)
	{
		//out2 << LabelMat.at<int>(i, 0) << endl;
		if (LabelMat.at<int>(i, 0) == 1)
		{
			out2 << "0 0" << endl;
		}
		if (LabelMat.at<int>(i, 0) == 2)
		{
			out2 << "0 1" << endl;
		}
	}
	out2.close();
}

void train()
{
	GetFeatureAndLabel(trPosPath, trNegPath, trPosName, trNegName, trFeatureMat, trLabelMat);

	//cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	//svm->setType(cv::ml::SVM::C_SVC);
	//svm->setKernel(cv::ml::SVM::LINEAR);
	//svm->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON));
	//svm->train(trFeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, trLabelMat);
	//svm->save(classifier_name);


	//cv::Ptr<cv::ml::KNearest> KNearest = cv::ml::KNearest::create();
	//KNearest->setDefaultK(2);
	//KNearest->setIsClassifier(true);
	//KNearest->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
	//cout << "KNearest Training Start..." << endl;
	//KNearest->train(trFeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, trLabelMat);
	////  KNearest.train_auto(sampleFeatureMat, sampleFeatureMat, Mat(), Mat(), params);
	//KNearest->save(classifier_name);
	//cout << "KNearest Training Complete" << endl;

	//cv::Ptr<cv::ml::RTrees> RTrees = cv::ml::RTrees::create();
	//RTrees->setMaxDepth(10);
	//RTrees->setMinSampleCount(10);
	//RTrees->setRegressionAccuracy(0);
	//RTrees->setUseSurrogates(false);
	//RTrees->setMaxCategories(2);
	//RTrees->setPriors(Mat());
	//RTrees->setCalculateVarImportance(true);
	//RTrees->setActiveVarCount(4);
	//RTrees->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + (0.01f > 0 ? cv::TermCriteria::EPS : 0), 100, 0.01f));
	//RTrees->train(trFeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, trLabelMat);
	//RTrees->save(classifier_name);

	//cv::Ptr<cv::ml::NormalBayesClassifier> NormalBayesClassifier = cv::ml::NormalBayesClassifier::create();
	//NormalBayesClassifier->train(trFeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, trLabelMat);
	//NormalBayesClassifier->save(classifier_name);

	//cv::Ptr<cv::ml::Boost> Boost = cv::ml::Boost::create();
	//Boost->setBoostType(cv::ml::Boost::DISCRETE);
	//Boost->setWeakCount(100);
	//Boost->setWeightTrimRate(0.95);
	//Boost->setMaxDepth(5);
	//Boost->setUseSurrogates(false);
	//Boost->setPriors(Mat());
	//Boost->train(trFeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, trLabelMat);
	//Boost->save(classifier_name);

	int in_num = trFeatureMat.cols;
	Mat ann_label = Mat::zeros(trFeatureMat.rows, 2, CV_32FC1);
	for (int i = 0; i < ann_label.rows; i++)
	{
		for (int j = 0; j < ann_label.rows; j++)
		{
			if (trLabelMat.at<int>(i, 0) > 0)
			{
				ann_label.at<int>(i, 1) = 1;
			}
			else if (trLabelMat.at<int>(i, 0) < 0)
			{
				ann_label.at<int>(i, 0) = 1;
			}
		}
	}
	for (int i = 0; i < ann_label.rows; i++)
		cout << ann_label.at<float>(i, 0) << " " << ann_label.at<float>(i, 1) << endl;

	cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();
	cv::Mat layerSizes = (cv::Mat_<int>(1, 5) << in_num, in_num, in_num/2, in_num/4, 2);
	ann->setLayerSizes(layerSizes);
	ann->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.001, 0.1);
	ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
	ann->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 10000, 0.0001));
	ann->train(trFeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, ann_label);
	ann->save(classifier_name);
}

void accuracy()
{
	GetFeatureAndLabel(tePosPath, teNegPath, tePosName, teNegName, teFeatureMat, teLabelMat);

	vector<string> TePosSamples;
	vector<string> TeNegSamples;
	GetPath(tePosPath, teNegPath, tePosName, teNegName, TePosSamples, TeNegSamples);
	int posSampleNum = TePosSamples.size();
	int negSampleNum = TeNegSamples.size();

	Mat result = Mat::zeros(posSampleNum + negSampleNum, 1, CV_32S);

	//cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(classifier_name);
	//svm->predict(teFeatureMat, result);

	//cv::Ptr<cv::ml::KNearest> svm = cv::Algorithm::load<cv::ml::KNearest>(classifier_name);
	//svm->predict(teFeatureMat, result);

	//cv::Ptr<cv::ml::RTrees> svm = cv::Algorithm::load<cv::ml::RTrees>(classifier_name);
	//svm->predict(teFeatureMat, result);

	//cv::Ptr<cv::ml::NormalBayesClassifier> svm = cv::Algorithm::load<cv::ml::NormalBayesClassifier>(classifier_name);
	//svm->predict(teFeatureMat, result);

	//cv::Ptr<cv::ml::Boost> svm = cv::Algorithm::load<cv::ml::Boost>(classifier_name);
	//svm->predict(teFeatureMat, result);
	//cout << result.rows << endl;

	cv::Ptr<cv::ml::ANN_MLP> svm = cv::ml::ANN_MLP::load(classifier_name);


	for (int i = 0; i < posSampleNum; i++)
	{
		Mat inputImg = cv::imread(TePosSamples[i]);
		//std::cout << "processing " << i << "/" << posSampleNum << " " << posSamples[i] << endl;
		Mat trainImg = Mat(dsize, CV_32S);
		resize(inputImg, trainImg, dsize);
		cvtColor(trainImg, trainImg, CV_BGR2GRAY);

		vector<float> descriptor;
		descriptor.clear();
		my_feature(trainImg, descriptor);
		Mat fea = Mat::zeros(1, descriptor.size(), CV_32FC1);;
		my_feature2(trainImg, fea);

		Mat dst2;
		svm->predict(fea, dst2);
		double maxVal = 0;
		cv::Point maxLoc;
		minMaxLoc(dst2, NULL, &maxVal, NULL, &maxLoc);
		cout << maxLoc.x << endl;
	}


	//int tp = 0, fp = 0, fn = 0, tn = 0;
	//for (int i = 0; i < result.rows; i++)
	//{
	//	int truth = teLabelMat.at<int>(i, 0);
	//	int predict = result.at<float>(i, 0);
	//	//if (predict > 0)
	//	//	predict = 1;
	//	//else
	//	//	predict = -1;
	//	cout << truth << " " << predict << endl;
 //
	//	if (truth == 1 && predict == 1)
	//		tp++;
	//	else if (truth == 1 && predict == -1)
	//		fn++;
	//	else if (truth == -1 && predict == 1)
	//		fp++;
	//	else if (truth == -1 && predict == -1)
	//		tn++;
	//}
	//cout << "tp num:" << tp << endl;
	//cout << "fp num:" << fp << endl;
	//cout << "fn num:" << fn << endl;
	//cout << "tn num:" << tn << endl;

	//cout << "pos num:" << tp + fn << endl;
	//cout << "neg num:" << fp + tn << endl;
	//float accuracy = (float)(tp + tn) / (float)(tp+fp+fn+tn);
	//float precision = (float)tp / (float)(tp+fp);
	//float recall = (float)tp / (float)(tp+fn);
	//cout << "acc: " << accuracy << endl;
	//cout << "prec: " << precision << endl;
	//cout << "recall: " << recall << endl;
}

void save()
{
	save_data("tr_data.txt", "tr_label.txt", trPosPath, trNegPath);
	save_data("te_data.txt", "te_label.txt", tePosPath, teNegPath);
	cout << "save finish!" << endl;
}

float getDistance(cv::Point A,cv::Point B)
{
	float dis = (A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y);
	//dis =sqrt(dis);
	return dis;
}

bool rectA_intersect_rectB(cv::Rect rectA, cv::Rect rectB)
{
	if (rectA.x > rectB.x + rectB.width) { return false; }
	if (rectA.y > rectB.y + rectB.height) { return false; }
	if ((rectA.x + rectA.width) < rectB.x) { return false; }
	if ((rectA.y + rectA.height) < rectB.y) { return false; }

	float colInt = min(rectA.x + rectA.width, rectB.x + rectB.width) - max(rectA.x, rectB.x);
	float rowInt = min(rectA.y + rectA.height, rectB.y + rectB.height) - max(rectA.y, rectB.y);
	float intersection = colInt * rowInt;
	float areaA = rectA.width * rectA.height;
	float areaB = rectB.width * rectB.height;
	float intersectionPercent = intersection / (areaA + areaB - intersection);

	if ((0 < intersectionPercent) && (intersectionPercent < 1) && (intersection != areaA) && (intersection != areaB))
	{
		return true;
	}
	return false;
}

void MyGammaCorrection(Mat src, Mat& dst, float fGamma)
{
	CV_Assert(src.data);
	CV_Assert(src.depth() != sizeof(uchar));

	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}

	dst = src.clone();
	const int channels = dst.channels();
	switch (channels)
	{
	case 1:
	{

		cv::MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			*it = lut[(*it)];

		break;
	}
	case 3:
	{

		cv::MatIterator_<cv::Vec3b> it, end;
		for (it = dst.begin<cv::Vec3b>(), end = dst.end<cv::Vec3b>(); it != end; it++)
		{
			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}

		break;

	}
	}
}


Mat get_sr(Mat I)
{
	//Mat I = imread(path);
	//if (I.empty())
	//	return -1;
	if (I.channels() == 3)
		cvtColor(I, I, CV_BGR2GRAY);
	Mat planes[] = { cv::Mat_<float>(I), Mat::zeros(I.size(), CV_32F) };
	Mat complexI; //复数矩阵
	merge(planes, 2, complexI); //把单通道矩阵组合成复数形式的双通道矩阵
	dft(complexI, complexI);  // 使用离散傅立叶变换

	//对复数矩阵进行处理，方法为谱残差
	Mat mag, pha, mag_mean;
	Mat Re, Im;
	split(complexI, planes); //分离复数到实部和虚部
	Re = planes[0]; //实部
	Im = planes[1]; //虚部
	magnitude(Re, Im, mag); //计算幅值
	phase(Re, Im, pha); //计算相角

	float *pre, *pim, *pm, *pp;
	//对幅值进行对数化
	for (int i = 0; i < mag.rows; i++)
	{
		pm = mag.ptr<float>(i);
		for (int j = 0; j < mag.cols; j++)
		{
			*pm = log(*pm);
			pm++;
		}
	}
	blur(mag, mag_mean, cv::Size(5, 5)); //对数谱的均值滤波
	mag = mag - mag_mean; //求取对数频谱残差
	//把对数谱残差的幅值和相角划归到复数形式
	for (int i = 0; i < mag.rows; i++)
	{
		pre = Re.ptr<float>(i);
		pim = Im.ptr<float>(i);
		pm = mag.ptr<float>(i);
		pp = pha.ptr<float>(i);
		for (int j = 0; j < mag.cols; j++)
		{
			*pm = exp(*pm);
			*pre = *pm * cos(*pp);
			*pim = *pm * sin(*pp);
			pre++;
			pim++;
			pm++;
			pp++;
		}
	}
	Mat planes1[] = { cv::Mat_<float>(Re), cv::Mat_<float>(Im) };

	merge(planes1, 2, complexI); //重新整合实部和虚部组成双通道形式的复数矩阵
	idft(complexI, complexI, cv::DFT_SCALE); // 傅立叶反变换
	split(complexI, planes); //分离复数到实部和虚部
	Re = planes[0];
	Im = planes[1];
	magnitude(Re, Im, mag); //计算幅值和相角
	for (int i = 0; i < mag.rows; i++)
	{
		pm = mag.ptr<float>(i);
		for (int j = 0; j < mag.cols; j++)
		{
			*pm = (*pm) * (*pm);
			pm++;
		}
	}
	GaussianBlur(mag, mag, cv::Size(7, 7), 2.5, 2.5);
	Mat invDFT, invDFTcvt;
	normalize(mag, invDFT, 0, 255, cv::NORM_MINMAX); //归一化到[0,255]供显示
	invDFT.convertTo(invDFTcvt, CV_8U); //转化成CV_8U型
	//imshow("SpectualResidual", invDFTcvt);
	//imshow("Original Image", I);
	return invDFTcvt;
}

cv::Rect mergeRects(vector<cv::Rect> rects)
{
	int tlx = 500, tly = 500, brx = 0, bry = 0;
	for (size_t k = 0; k < rects.size(); k++)
	{
		if (rects[k].tl().x < tlx)
		{
			tlx = rects[k].tl().x;
		}
		if (rects[k].tl().y < tly)
		{
			tly = rects[k].tl().y;
		}

		if (rects[k].br().x > brx)
		{
			brx = rects[k].br().x;
		}
		if (rects[k].br().y > bry)
		{
			bry = rects[k].br().y;
		}
	}
	cv::Rect NewRect(tlx, tly, brx - tlx, bry - tly);
	return NewRect;
}

void detect()
{
	vector<string> posSamples;
	vector<string> negSamples;
	GetPath(trPosPath, trNegPath, trPosName, trNegName, posSamples, negSamples);
	int posSampleNum = posSamples.size();
	int negSampleNum = negSamples.size();
	Mat tmp1 = cv::imread(posSamples[0]);
	Mat tmp2 = Mat(dsize, CV_32S);
	resize(tmp1, tmp2, dsize);
	cvtColor(tmp2, tmp2, CV_BGR2GRAY);
	vector<float> tmp;
	tmp.clear();
	my_feature(tmp2, tmp);
	descriptordim = tmp.size();
	cout << "dim is: " << descriptordim << endl;

	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(classifier_name);

	for (int i=1;i<=93;i++)
	{
		string name = detestTestName + std::to_string(i) + ".jpg";
		Mat src = cv::imread(name);
		if (src.empty())
			continue;


		//////////////////////////////////////////////////////////////////////hog-svm
		cv::HOGDescriptor hog("hogsvm_64_16.xml");
		vector< cv::Rect> detections;
		vector< double> foundWeights;
		Mat show0 = src.clone();
		hog.detectMultiScale(src, detections, foundWeights);
		for (size_t j = 0; j < detections.size(); j++)
		{
			rectangle(show0, detections[j], cv::Scalar(255,255,255), src.cols / 400 + 1);
		}
		imshow("传统HOGSVM", show0);


		////////////////////////////////////////////////////////////////////fhog-svm
		Mat mat1 = src.clone();
		cvtColor(mat1, mat1, cv::COLOR_RGB2GRAY);
		vector<cv::Rect> rects;
		vector<cv::Point> points;
		rects.clear();
		for (int y = 0; y < src.cols - winsize + 1; y = y + stride)
		{
			for (int x = 0; x < src.rows - winsize + 1; x = x + stride)
			{
				cv::Rect rectTmp(x,y, winsize, winsize);
				Mat roi = mat1(rectTmp).clone();
				resize(roi, roi, dsize);
				Mat fea = Mat::zeros(1, descriptordim, CV_32FC1);;
				my_feature2(roi, fea);
				float respose = svm->predict(fea);
				//cout << respose<<endl;
				if (respose == 1)
				{
					cv::Point centerTmp(rectTmp.x+ winsize/2,rectTmp.y+ winsize/2);
					points.push_back(centerTmp);
					rects.push_back(rectTmp);
					//cout << rectTmp.x<<"  "<<rectTmp.y << endl;
				}
			}
		}
		Mat show1 = src.clone();
		for (int i = 0; i < rects.size(); i++)
		{
			//circle(show,points[i],3,cv::Scalar(0,255,0));
			rectangle(show1, rects[i], cv::Scalar(0, 255, 0));
		}
		imshow("只用fhog", show1);


		//////////////////////////////////////////////////////////////////////phase-saliency
		Mat mat2 = src.clone();
		cvtColor(mat2, mat2, cv::COLOR_RGB2GRAY);
		MyGammaCorrection(mat2, mat2, 5);
		//imshow("gamma", mat2);
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;
		Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);
		Sobel(src, grad_y, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, mat2);
		//imshow("sobel", mat2);
		Mat sr = get_sr(mat2);
		//imshow("sr", sr);
		Mat bin;
		threshold(sr, bin, 0, 255, cv::THRESH_OTSU);
		Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
		morphologyEx(bin, bin, cv::MORPH_CLOSE, element);
		morphologyEx(bin, bin, cv::MORPH_DILATE, element);
		vector<vector<cv::Point> >contours;
		vector<cv::Vec4i> hierarchy;
		findContours(bin, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
		vector<vector<cv::Point> > contoursNew;
		for (int i = 0; i < contours.size(); i++)
		{
			//cout << contourArea(contours[i]) << endl;
			int areaTmp = contourArea(contours[i]);
			if (areaTmp > 400 && areaTmp < 8000)
			{
				contoursNew.push_back(contours[i]);
			}
		}
		//drawContours(input, contoursNew, -1, Scalar(0,255,0),-1);
		vector<cv::Rect> boundRect(contoursNew.size());
		Mat show2 = src.clone();
		for (int i = 0; i < contoursNew.size(); i++)
		{
			cv::Rect rectTmp = boundingRect(contoursNew[i]);
			boundRect.push_back(rectTmp);
			rectangle(show2, rectTmp, cv::Scalar(255, 0, 0), 1);
		}
		imshow("只用频域",show2);


		//////////////////////////////////////////////////////////////////////交并处理
		cv::Rect NewRect = mergeRects(rects);
		int newrect_area = NewRect.area();

		vector<cv::Rect> finalrects;
		finalrects.push_back(NewRect);
		for (int i = 0; i < boundRect.size(); i++)
		{
			if (rectA_intersect_rectB(boundRect[i], NewRect))
			{
				cv::Rect intersect = boundRect[i] & NewRect;
				int in_area = intersect.area();
				float rate = in_area / intersect.area();
				if (rate > 0)
				{
					finalrects.push_back(boundRect[i]);
				}
			}
		}

		cv::Rect NewRect2 = mergeRects(finalrects);
		//cv::Rect NewRect3(((NewRect2.tl().x+ NewRect2.br().x)/2)-40, ((NewRect2.tl().y + NewRect2.br().y) / 2) - 40, 80, 80);
		
		Mat show3 = src.clone();
		rectangle(show3, NewRect2, cv::Scalar(0, 0, 255));
		imshow("交并处理", show3);


		////////////////////////////////////////////////////////////////////再次判定


		cv::waitKey();
	}
	std::cout << "detect finish" << endl;
}
