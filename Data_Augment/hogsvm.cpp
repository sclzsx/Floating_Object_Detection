//#include "opencv2/imgproc.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/ml.hpp"
//#include "opencv2/objdetect.hpp"
//#include <iostream>
//#include <time.h>
//#include <io.h> 
//#include <stdio.h>  
//#include <string>
//
//using std::cout;
//using std::endl;
//using std::string;
//using std::vector;
//
//using cv::Ptr;
//using cv::ml::SVM;
//using cv::Mat;
//
//string pos_dir = "E:\\DataSets\\litter\\2class_unresize\\pos\\";
//string neg_dir = "E:\\DataSets\\litter\\2class_unresize\\neg\\";
//string test_dir = "E:\\DataSets\\litter\\src_V2\\detect\\";
//string obj_det_filename = "hogsvm_64_16.xml";
//
//string videofilename = "";
//int detector_width = 64;
//int detector_height = 64;
//bool test_detector = 1;
//bool train_twice = 0;
//bool visualization = 0;
//bool flip_samples = 0;
//
//vector< float > get_svm_detector(const Ptr< SVM >& svm)
//{
//	// get the support vectors
//	Mat sv = svm->getSupportVectors();
//	const int sv_total = sv.rows;
//	// get the decision function
//	Mat alpha, svidx;
//	double rho = svm->getDecisionFunction(0, alpha, svidx);
//	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
//	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
//		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
//	CV_Assert(sv.type() == CV_32F);
//	vector< float > hog_detector(sv.cols + 1);
//	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
//	hog_detector[sv.cols] = (float)-rho;
//	return hog_detector;
//}
//
//void convert_to_ml(const vector< Mat > & train_samples, Mat& trainData)
//{
//	//--Convert data
//	const int rows = (int)train_samples.size();
//	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
//	Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
//	trainData = Mat(rows, cols, CV_32FC1);
//	for (size_t i = 0; i < train_samples.size(); ++i)
//	{
//		CV_Assert(train_samples[i].cols == 1 || train_samples[i].rows == 1);
//		if (train_samples[i].cols == 1)
//		{
//			transpose(train_samples[i], tmp);
//			tmp.copyTo(trainData.row((int)i));
//		}
//		else if (train_samples[i].rows == 1)
//		{
//			train_samples[i].copyTo(trainData.row((int)i));
//		}
//	}
//}
//
//void load_images(const string & dirname, vector< Mat > & img_lst, bool showImages = false)
//{
//	vector< string > files;
//	cv::glob(dirname, files);
//	for (size_t i = 0; i < files.size(); ++i)
//	{
//		Mat img = cv::imread(files[i]); // load the image
//		if (img.empty())            // invalid image, skip it.
//		{
//			cout << files[i] << " is invalid!" << endl;
//			continue;
//		}
//		if (showImages)
//		{
//			imshow("image", img);
//			cv::waitKey(1);
//		}
//		img_lst.push_back(img);
//	}
//}
//
//void sample_neg(const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const cv::Size & size)
//{
//	cv::Rect box;
//	box.width = size.width;
//	box.height = size.height;
//	const int size_x = box.width;
//	const int size_y = box.height;
//	srand((unsigned int)time(NULL));
//	for (size_t i = 0; i < full_neg_lst.size(); i++)
//		if (full_neg_lst[i].cols > box.width && full_neg_lst[i].rows > box.height)
//		{
//			box.x = rand() % (full_neg_lst[i].cols - size_x);
//			box.y = rand() % (full_neg_lst[i].rows - size_y);
//			Mat roi = full_neg_lst[i](box);
//			neg_lst.push_back(roi.clone());
//		}
//}
//
//void computeHOGs(const cv::Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst, bool use_flip)
//{
//	cv::HOGDescriptor hog;
//	hog.winSize = wsize;
//	Mat gray;
//	vector< float > descriptors;
//	for (size_t i = 0; i < img_lst.size(); i++)
//	{
//		if (img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height)
//		{
//			cv::Rect r = cv::Rect((img_lst[i].cols - wsize.width) / 2,
//				(img_lst[i].rows - wsize.height) / 2,
//				wsize.width,
//				wsize.height);
//			cvtColor(img_lst[i](r), gray, cv::COLOR_BGR2GRAY);
//			hog.compute(gray, descriptors, cv::Size(8, 8), cv::Size(0, 0));
//			gradient_lst.push_back(Mat(descriptors).clone());
//			if (use_flip)
//			{
//				flip(gray, gray, 1);
//				hog.compute(gray, descriptors, cv::Size(8, 8), cv::Size(0, 0));
//				gradient_lst.push_back(Mat(descriptors).clone());
//			}
//		}
//	}
//}
//
//void test_trained_detector(string obj_det_filename, string test_dir, string videofilename)
//{
//	cout << "Testing trained detector..." << endl;
//	cv::HOGDescriptor hog;
//	hog.load(obj_det_filename);
//	vector< string > files;
//	cv::glob(test_dir, files);
//	int delay = 0;
//	cv::VideoCapture cap;
//	if (videofilename != "")
//	{
//		if (videofilename.size() == 1 && isdigit(videofilename[0]))
//			cap.open(videofilename[0] - '0');
//		else
//			cap.open(videofilename);
//	}
//	obj_det_filename = "testing " + obj_det_filename;
//
//	cv::namedWindow(obj_det_filename, cv::WINDOW_NORMAL);
//
//	//cv::VideoWriter writer("video_out.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 3, Size(350, 350));
//
//	for (size_t i = 0;; i++)
//	{
//		Mat img;
//		if (cap.isOpened())
//		{
//			cap >> img;
//			delay = 30;
//		}
//		else if (i < files.size())
//		{
//			img = cv::imread(files[i]);
//		}
//		if (img.empty())
//		{
//			return;
//		}
//		vector< cv::Rect > detections;
//		vector< double > foundWeights;
//		hog.detectMultiScale(img, detections, foundWeights);
//		for (size_t j = 0; j < detections.size(); j++)
//		{
//			cv::Scalar color = cv::Scalar(0, foundWeights[j] * foundWeights[j] * 200, 0);
//			rectangle(img, detections[j], color, img.cols / 400 + 1);
//		}
//		//writer.write(img);
//		imshow(obj_det_filename, img);
//		if (cv::waitKey(delay) == 27)
//		{
//			return;
//		}
//	}
//}
//
//void getFiles(std::string path, vector<std::string>& files)
//{
//	intptr_t   hFile = 0;
//	struct _finddata_t fileinfo;
//	std::string p;
//	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
//	{
//
//		do
//		{
//			if ((fileinfo.attrib &  _A_SUBDIR))
//			{
//				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
//					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
//			}
//			else
//			{
//				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
//			}
//
//		} while (_findnext(hFile, &fileinfo) == 0);
//
//		_findclose(hFile);
//	}
//}
//
//void experiment()
//{
//	cv::HOGDescriptor hog;
//	hog.load(obj_det_filename);
//	std::string filePath = test_dir;
//	vector<std::string> files;
//
//	getFiles(filePath, files);
//	int number = files.size();
//	for (int i = 0; i < number; i++)
//	{
//		Mat  img = cv::imread(files[i].c_str());
//		vector< cv::Rect > detections;
//		vector< double > foundWeights;
//		hog.detectMultiScale(img, detections, foundWeights);
//		for (size_t j = 0; j < detections.size(); j++)
//		{
//			cv::Scalar color = cv::Scalar(255, foundWeights[j] * foundWeights[j] * 200, 0);
//			rectangle(img, detections[j], color, img.cols / 400 + 1);
//		}
//		cv::imshow("rect", img);
//		cv::waitKey();
//	}
//}
//
//void train_hog_svm_detect()
//{
//	vector< Mat > pos_lst, full_neg_lst, neg_lst, gradient_lst;
//	vector< int > labels;
//	cout << "Positive images are being loaded...";
//	load_images(pos_dir, pos_lst, visualization);
//	if (pos_lst.size() > 0)
//	{
//		cout << "...[done]" << endl;
//	}
//	cv::Size pos_image_size = pos_lst[0].size();
//	if (detector_width && detector_height)
//	{
//		pos_image_size = cv::Size(detector_width, detector_height);
//	}
//	else
//	{
//		for (size_t i = 0; i < pos_lst.size(); ++i)
//		{
//			if (pos_lst[i].size() != pos_image_size)
//			{
//				cout << "All positive images should be same size!" << endl;
//			}
//		}
//		pos_image_size = pos_image_size / 8 * 8;
//	}
//
//	cout << "Negative images are being loaded...";
//	load_images(neg_dir, full_neg_lst, false);
//	sample_neg(full_neg_lst, neg_lst, pos_image_size);
//	cout << "...[done]" << endl;
//
//	cout << "Histogram of Gradients are being calculated for positive images...";
//	computeHOGs(pos_image_size, pos_lst, gradient_lst, flip_samples);
//	size_t positive_count = gradient_lst.size();
//	labels.assign(positive_count, +1);
//	cout << "...[done] ( positive count : " << positive_count << " )" << endl;
//
//	cout << "Histogram of Gradients are being calculated for negative images...";
//	computeHOGs(pos_image_size, neg_lst, gradient_lst, flip_samples);
//	size_t negative_count = gradient_lst.size() - positive_count;
//	labels.insert(labels.end(), negative_count, -1);
//	CV_Assert(positive_count < labels.size());
//	cout << "...[done] ( negative count : " << negative_count << " )" << endl;
//	Mat train_data;
//	convert_to_ml(gradient_lst, train_data);
//	cout << "Training SVM...";
//	Ptr< SVM > svm = SVM::create();
//	/* Default values to train SVM */
//	svm->setCoef0(0.0);
//	svm->setDegree(3);
//	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-3));
//	svm->setGamma(0);
//	svm->setKernel(SVM::LINEAR);
//	svm->setNu(0.5);
//	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
//	svm->setC(0.01); // From paper, soft classifier
//	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
//	svm->train(train_data, cv::ml::ROW_SAMPLE, labels);
//	cout << "...[done]" << endl;
//	if (train_twice)
//	{
//		cout << "Testing trained detector on negative images. This may take a few minutes...";
//		cv::HOGDescriptor my_hog;
//		my_hog.winSize = pos_image_size;
//		// Set the trained svm to my_hog
//		my_hog.setSVMDetector(get_svm_detector(svm));
//		vector<cv::Rect > detections;
//		vector< double > foundWeights;
//		for (size_t i = 0; i < full_neg_lst.size(); i++)
//		{
//			if (full_neg_lst[i].cols >= pos_image_size.width && full_neg_lst[i].rows >= pos_image_size.height)
//				my_hog.detectMultiScale(full_neg_lst[i], detections, foundWeights);
//			else
//				detections.clear();
//			for (size_t j = 0; j < detections.size(); j++)
//			{
//				Mat detection = full_neg_lst[i](detections[j]).clone();
//				resize(detection, detection, pos_image_size, 0, 0, cv::INTER_LINEAR_EXACT);
//				neg_lst.push_back(detection);
//			}
//			if (visualization)
//			{
//				for (size_t j = 0; j < detections.size(); j++)
//				{
//					rectangle(full_neg_lst[i], detections[j], cv::Scalar(0, 255, 0), 2);
//				}
//				imshow("testing trained detector on negative images", full_neg_lst[i]);
//				cv::waitKey(5);
//			}
//		}
//		cout << "...[done]" << endl;
//		gradient_lst.clear();
//		cout << "Histogram of Gradients are being calculated for positive images...";
//		computeHOGs(pos_image_size, pos_lst, gradient_lst, flip_samples);
//		positive_count = gradient_lst.size();
//		cout << "...[done] ( positive count : " << positive_count << " )" << endl;
//		cout << "Histogram of Gradients are being calculated for negative images...";
//		computeHOGs(pos_image_size, neg_lst, gradient_lst, flip_samples);
//		negative_count = gradient_lst.size() - positive_count;
//		cout << "...[done] ( negative count : " << negative_count << " )" << endl;
//		labels.clear();
//		labels.assign(positive_count, +1);
//		labels.insert(labels.end(), negative_count, -1);
//		cout << "Training SVM again...";
//		convert_to_ml(gradient_lst, train_data);
//		svm->train(train_data, cv::ml::ROW_SAMPLE, labels);
//		cout << "...[done]" << endl;
//	}
//	cv::HOGDescriptor hog;
//	hog.winSize = pos_image_size;
//	hog.setSVMDetector(get_svm_detector(svm));
//	hog.save(obj_det_filename);
//	//test_trained_detector(obj_det_filename, test_dir, videofilename);
//}
//
//int main()
//{
//	train_hog_svm_detect();
//	system("pause");
//	return 0;
//}