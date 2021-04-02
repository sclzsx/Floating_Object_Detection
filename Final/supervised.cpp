#include "supervised.h"

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

//void svm_train(string pos_path, string neg_path, int feature_type, string &xml_filename)
//{
//	vector< Mat > pos_mats, neg_mats;
//	load_images(pos_path, pos_mats, 1);
//	load_images(neg_path, neg_mats, 1);
//	int pos_num = pos_mats.size();
//	int neg_num = neg_mats.size();
//	int total_num = pos_num + neg_num;
//	cout << "pos num: " << pos_num << endl;
//	cout << "neg num: " << neg_num << endl;
//
//	vector<float> feature_tmp;
//	if (feature_type == fhog_type_)
//		get_fhog_feature_vector(pos_mats[0], feature_tmp);
//	else if (feature_type == hog_type_)
//		get_hog_feature_vector(pos_mats[0], feature_tmp);
//	else if (feature_type == lbp_type_)
//		get_lbp_feature_vector(pos_mats[0], feature_tmp);
//	else if (feature_type == glcm_type_)
//		get_glcm_feature_vector(pos_mats[0], feature_tmp);
//	else if (feature_type == fhog_glcm_type_)
//		get_fhog_glcm_feature_vector(pos_mats[0], feature_tmp);
//
//	int dim = 0;
//	dim = feature_tmp.size();
//	cout << "dim is: " << dim << endl;
//	Mat FeatureMat;
//	Mat LabelMat;
//	FeatureMat = Mat::zeros(total_num, dim, CV_32FC1);
//	LabelMat = Mat::zeros(total_num, 1, CV_32S);
//	for (int i = 0; i < pos_num; i++)
//	{
//		vector<float> descriptor;
//		if (feature_type == fhog_type_)
//			get_fhog_feature_vector(pos_mats[i], descriptor);
//		else if (feature_type == hog_type_)
//			get_hog_feature_vector(pos_mats[i], descriptor);
//		else if (feature_type == lbp_type_)
//			get_lbp_feature_vector(pos_mats[i], descriptor);
//		else if (feature_type == glcm_type_)
//			get_glcm_feature_vector(pos_mats[i], descriptor);
//		else if (feature_type == fhog_glcm_type_)
//			get_fhog_glcm_feature_vector(pos_mats[i], descriptor);
//
//		for (int j = 0; j < dim; j++)
//		{
//			FeatureMat.at<float>(i, j) = descriptor[j];
//		}
//		LabelMat.at<int>(i, 0) = 1;
//	}
//	for (int i = 0; i < neg_num; i++)
//	{
//		vector<float> descriptor;
//		if (feature_type == fhog_type_)
//			get_fhog_feature_vector(neg_mats[i], descriptor);
//		else if (feature_type == hog_type_)
//			get_hog_feature_vector(neg_mats[i], descriptor);
//		else if (feature_type == lbp_type_)
//			get_lbp_feature_vector(neg_mats[i], descriptor);
//		else if (feature_type == glcm_type_)
//			get_glcm_feature_vector(neg_mats[i], descriptor);
//		else if (feature_type == fhog_glcm_type_)
//			get_fhog_glcm_feature_vector(neg_mats[i], descriptor);
//
//		for (int j = 0; j < dim; j++)
//		{
//			FeatureMat.at<float>(i + pos_num, j) = descriptor[j];
//		}
//		LabelMat.at<int>(i + pos_num, 0) = -1;
//	}
//
//
//	cv::Ptr<cv::ml::SVM> svm_tmp = cv::ml::SVM::create();
//	svm_tmp->setType(cv::ml::SVM::C_SVC);
//	svm_tmp->setKernel(cv::ml::SVM::LINEAR);
//	svm_tmp->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON));
//	cout << "Training Start..." << endl;
//	svm_tmp->train(FeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, LabelMat);
//
//	xml_filename = "featype_" + to_string(feature_type) + "_svm.xml";
//	cout << xml_filename << endl;
//	svm_tmp->save(xml_filename);
//	cout << "Training Complete" << endl;
//}
//
//void svm_classify(Mat input, int feature_type, cv::Ptr<cv::ml::SVM> input_svm, vector<cv::Rect> &output_rects)
//{
//	for (int y = 0; y < input.cols - winsize + 1; y = y + stride)
//	{
//		for (int x = 0; x < input.rows - winsize + 1; x = x + stride)
//		{
//			cv::Rect rectTmp(x, y, winsize, winsize);
//			Mat roi = input(rectTmp).clone();
//
//			vector<float> feature_vec;
//			if (feature_type == fhog_type_)
//				get_fhog_feature_vector(roi, feature_vec);
//			else if (feature_type == hog_type_)
//				get_hog_feature_vector(roi, feature_vec);
//			else if (feature_type == lbp_type_)
//				get_lbp_feature_vector(roi, feature_vec);
//			else if (feature_type == glcm_type_)
//				get_glcm_feature_vector(roi, feature_vec);
//			else if (feature_type == fhog_glcm_type_)
//				get_fhog_glcm_feature_vector(roi, feature_vec);
//
//			int dim = feature_vec.size();
//			//cout << dim << endl;
//			Mat feature_mat = Mat::zeros(1, dim, CV_32FC1);
//			feature_vec2mat(feature_vec, feature_mat);
//			float response = input_svm->predict(feature_mat);
//			if (response == 1)
//			{
//				output_rects.push_back(rectTmp);
//			}
//		}
//	}
//}
//
//
//


void train(string pos_path, string neg_path, int feature_type, int classifier_type, string &xml_filename)
{
	xml_filename = info + "featype_" + to_string(feature_type) + "_classtype_" + to_string(classifier_type) + "_.xml";
	cout << "Training for " << xml_filename << "......";
	vector< Mat > pos_mats, neg_mats;
	load_images(pos_path, pos_mats, 1);
	load_images(neg_path, neg_mats, 1);
	int pos_num = pos_mats.size();
	int neg_num = neg_mats.size();
	int total_num = pos_num + neg_num;
	//cout << "pos num: " << pos_num << endl;
	//cout << "neg num: " << neg_num << endl;

	vector<float> feature_tmp;
	if (feature_type == fhog_type_)
		get_fhog_feature_vector(pos_mats[0], feature_tmp);
	else if (feature_type == hog_type_)
		get_hog_feature_vector(pos_mats[0], feature_tmp);
	else if (feature_type == lbp_type_)
		get_lbp_feature_vector(pos_mats[0], feature_tmp);
	else if (feature_type == glcm_type_)
		get_glcm_feature_vector(pos_mats[0], feature_tmp);
	else if (feature_type == fhog_glcm_type_)
		get_fhog_glcm_feature_vector(pos_mats[0], feature_tmp);

	int dim = 0;
	dim = feature_tmp.size();
	//cout << "dim is: " << dim << endl;
	Mat FeatureMat;
	Mat LabelMat;
	FeatureMat = Mat::zeros(total_num, dim, CV_32FC1);
	LabelMat = Mat::zeros(total_num, 1, CV_32S);
	for (int i = 0; i < pos_num; i++)
	{
		vector<float> descriptor;
		if (feature_type == fhog_type_)
			get_fhog_feature_vector(pos_mats[i], descriptor);
		else if (feature_type == hog_type_)
			get_hog_feature_vector(pos_mats[i], descriptor);
		else if (feature_type == lbp_type_)
			get_lbp_feature_vector(pos_mats[i], descriptor);
		else if (feature_type == glcm_type_)
			get_glcm_feature_vector(pos_mats[i], descriptor);
		else if (feature_type == fhog_glcm_type_)
			get_fhog_glcm_feature_vector(pos_mats[i], descriptor);

		for (int j = 0; j < dim; j++)
		{
			FeatureMat.at<float>(i, j) = descriptor[j];
		}
		LabelMat.at<int>(i, 0) = 1;
	}
	for (int i = 0; i < neg_num; i++)
	{
		vector<float> descriptor;
		if (feature_type == fhog_type_)
			get_fhog_feature_vector(neg_mats[i], descriptor);
		else if (feature_type == hog_type_)
			get_hog_feature_vector(neg_mats[i], descriptor);
		else if (feature_type == lbp_type_)
			get_lbp_feature_vector(neg_mats[i], descriptor);
		else if (feature_type == glcm_type_)
			get_glcm_feature_vector(neg_mats[i], descriptor);
		else if (feature_type == fhog_glcm_type_)
			get_fhog_glcm_feature_vector(neg_mats[i], descriptor);

		for (int j = 0; j < dim; j++)
		{
			FeatureMat.at<float>(i + pos_num, j) = descriptor[j];
		}
		LabelMat.at<int>(i + pos_num, 0) = -1;
	}

	if (classifier_type == svm_type_)
	{
		cv::Ptr<cv::ml::SVM> tmp = cv::ml::SVM::create();
		tmp->setType(cv::ml::SVM::C_SVC);
		tmp->setKernel(cv::ml::SVM::LINEAR);
		tmp->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON));
		tmp->train(FeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, LabelMat);
		tmp->save(xml_filename);

		//svm = cv::ml::SVM::load(xml_filename);
	}
	else if (classifier_type == knn_type_)
	{
		cv::Ptr<cv::ml::KNearest> tmp = cv::ml::KNearest::create();
		tmp->setDefaultK(2);
		tmp->setIsClassifier(true);
		tmp->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
		tmp->train(FeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, LabelMat);
		//  KNearest.train_auto(sampleFeatureMat, sampleFeatureMat, Mat(), Mat(), params);
		tmp->save(xml_filename);

		//knn = cv::Algorithm::load<cv::ml::KNearest>(xml_filename);
	}
	else if (classifier_type == rtrees_type_)
	{
		cv::Ptr<cv::ml::RTrees> tmp = cv::ml::RTrees::create();
		tmp->setMaxDepth(10);
		tmp->setMinSampleCount(10);
		tmp->setRegressionAccuracy(0);
		tmp->setUseSurrogates(false);
		tmp->setMaxCategories(2);
		tmp->setPriors(Mat());
		tmp->setCalculateVarImportance(true);
		tmp->setActiveVarCount(4);
		tmp->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + (0.01f > 0 ? cv::TermCriteria::EPS : 0), 100, 0.01f));
		tmp->train(FeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, LabelMat);
		tmp->save(xml_filename);

		//rtrees = cv::Algorithm::load<cv::ml::RTrees>(xml_filename);
	}
	else if (classifier_type == bayes_type_)
	{
		cv::Ptr<cv::ml::NormalBayesClassifier> NormalBayesClassifier = cv::ml::NormalBayesClassifier::create();
		NormalBayesClassifier->train(FeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, LabelMat);
		NormalBayesClassifier->save(xml_filename);

		//bayes = cv::Algorithm::load<cv::ml::NormalBayesClassifier>(xml_filename);
	}
	else if (classifier_type == adaboost_type_)
	{
		cv::Ptr<cv::ml::Boost> tmp = cv::ml::Boost::create();
		tmp->setBoostType(cv::ml::Boost::DISCRETE);
		tmp->setWeakCount(100);
		tmp->setWeightTrimRate(0.95);
		tmp->setMaxDepth(5);
		tmp->setUseSurrogates(false);
		tmp->setPriors(Mat());
		tmp->train(FeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, LabelMat);
		tmp->save(xml_filename);

		//adaboost = cv::Algorithm::load<cv::ml::Boost>(xml_filename);
	}
	else if (classifier_type == ann_type_)
	{
		int in_num = FeatureMat.cols;
		Mat ann_label = Mat::zeros(FeatureMat.rows, 2, CV_32FC1);
		for (int i = 0; i < ann_label.rows; i++)
		{
			for (int j = 0; j < ann_label.rows; j++)
			{
				if (LabelMat.at<int>(i, 0) > 0)
				{
					ann_label.at<int>(i, 1) = 1;
				}
				else if (LabelMat.at<int>(i, 0) < 0)
				{
					ann_label.at<int>(i, 0) = 1;
				}
			}
		}
		//for (int i = 0; i < ann_label.rows; i++)
		//	cout << ann_label.at<float>(i, 0) << " " << ann_label.at<float>(i, 1) << endl;
		cv::Ptr<cv::ml::ANN_MLP> tmp = cv::ml::ANN_MLP::create();
		cv::Mat layerSizes = (cv::Mat_<int>(1, 5) << in_num, in_num, in_num / 2, in_num / 4, 2);
		tmp->setLayerSizes(layerSizes);
		tmp->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.001, 0.1);
		tmp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
		tmp->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 10000, 0.0001));
		tmp->train(FeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, ann_label);
		tmp->save(xml_filename);

		//ann = cv::ml::ANN_MLP::load(xml_filename);
	}
	cout << "[done]" << endl;
}

void ann_classify(Mat input_mat, cv::Rect input_rect, int feature_type, cv::Ptr<cv::ml::ANN_MLP> input_classifier, vector<cv::Rect> &output_rects)
{
	vector<float> feature_vec;
	if (feature_type == fhog_type_)
		get_fhog_feature_vector(input_mat, feature_vec);
	else if (feature_type == hog_type_)
		get_hog_feature_vector(input_mat, feature_vec);
	else if (feature_type == lbp_type_)
		get_lbp_feature_vector(input_mat, feature_vec);
	else if (feature_type == glcm_type_)
		get_glcm_feature_vector(input_mat, feature_vec);
	else if (feature_type == fhog_glcm_type_)
		get_fhog_glcm_feature_vector(input_mat, feature_vec);
	int dim = feature_vec.size();
	//cout << dim << endl;
	Mat feature_mat = Mat::zeros(1, dim, CV_32FC1);
	feature_vec2mat(feature_vec, feature_mat);
	feature_vec.clear();

	Mat dst;
	input_classifier->predict(feature_mat, dst);
	double maxVal = 0;
	cv::Point maxLoc;
	minMaxLoc(dst, NULL, &maxVal, NULL, &maxLoc);
	float response = maxLoc.x;
	if (response == 1)
	{
		output_rects.push_back(input_rect);
	}
}

void svm_classify(Mat input_mat, cv::Rect input_rect, int feature_type, cv::Ptr<cv::ml::SVM> input_classifier, vector<cv::Rect> &output_rects)
{
	vector<float> feature_vec;
	if (feature_type == fhog_type_)
		get_fhog_feature_vector(input_mat, feature_vec);
	else if (feature_type == hog_type_)
		get_hog_feature_vector(input_mat, feature_vec);
	else if (feature_type == lbp_type_)
		get_lbp_feature_vector(input_mat, feature_vec);
	else if (feature_type == glcm_type_)
		get_glcm_feature_vector(input_mat, feature_vec);
	else if (feature_type == fhog_glcm_type_)
		get_fhog_glcm_feature_vector(input_mat, feature_vec);
	int dim = feature_vec.size();
	//cout << dim << endl;
	Mat feature_mat = Mat::zeros(1, dim, CV_32FC1);
	feature_vec2mat(feature_vec, feature_mat);
	feature_vec.clear();
	float response = input_classifier->predict(feature_mat);
	if (response == 1)
	{
		output_rects.push_back(input_rect);
	}
}

void knn_classify(Mat input_mat, cv::Rect input_rect, int feature_type, cv::Ptr<cv::ml::KNearest> input_classifier, vector<cv::Rect> &output_rects)
{
	vector<float> feature_vec;
	if (feature_type == fhog_type_)
		get_fhog_feature_vector(input_mat, feature_vec);
	else if (feature_type == hog_type_)
		get_hog_feature_vector(input_mat, feature_vec);
	else if (feature_type == lbp_type_)
		get_lbp_feature_vector(input_mat, feature_vec);
	else if (feature_type == glcm_type_)
		get_glcm_feature_vector(input_mat, feature_vec);
	else if (feature_type == fhog_glcm_type_)
		get_fhog_glcm_feature_vector(input_mat, feature_vec);
	int dim = feature_vec.size();
	//cout << dim << endl;
	Mat feature_mat = Mat::zeros(1, dim, CV_32FC1);
	feature_vec2mat(feature_vec, feature_mat);
	feature_vec.clear();
	float response = input_classifier->predict(feature_mat);
	if (response == 1)
	{
		output_rects.push_back(input_rect);
	}
}

void adaboost_classify(Mat input_mat, cv::Rect input_rect, int feature_type, cv::Ptr<cv::ml::Boost> input_classifier, vector<cv::Rect> &output_rects)
{
	vector<float> feature_vec;
	if (feature_type == fhog_type_)
		get_fhog_feature_vector(input_mat, feature_vec);
	else if (feature_type == hog_type_)
		get_hog_feature_vector(input_mat, feature_vec);
	else if (feature_type == lbp_type_)
		get_lbp_feature_vector(input_mat, feature_vec);
	else if (feature_type == glcm_type_)
		get_glcm_feature_vector(input_mat, feature_vec);
	else if (feature_type == fhog_glcm_type_)
		get_fhog_glcm_feature_vector(input_mat, feature_vec);
	int dim = feature_vec.size();
	//cout << dim << endl;
	Mat feature_mat = Mat::zeros(1, dim, CV_32FC1);
	feature_vec2mat(feature_vec, feature_mat);
	feature_vec.clear();
	float response = input_classifier->predict(feature_mat);
	if (response == 1)
	{
		output_rects.push_back(input_rect);
	}
}

void rtrees_classify(Mat input_mat, cv::Rect input_rect, int feature_type, cv::Ptr<cv::ml::RTrees> input_classifier, vector<cv::Rect> &output_rects)
{
	vector<float> feature_vec;
	if (feature_type == fhog_type_)
		get_fhog_feature_vector(input_mat, feature_vec);
	else if (feature_type == hog_type_)
		get_hog_feature_vector(input_mat, feature_vec);
	else if (feature_type == lbp_type_)
		get_lbp_feature_vector(input_mat, feature_vec);
	else if (feature_type == glcm_type_)
		get_glcm_feature_vector(input_mat, feature_vec);
	else if (feature_type == fhog_glcm_type_)
		get_fhog_glcm_feature_vector(input_mat, feature_vec);
	int dim = feature_vec.size();
	//cout << dim << endl;
	Mat feature_mat = Mat::zeros(1, dim, CV_32FC1);
	feature_vec2mat(feature_vec, feature_mat);
	feature_vec.clear();
	float response = input_classifier->predict(feature_mat);
	if (response == 1)
	{
		output_rects.push_back(input_rect);
	}
}

void bayes_classify(Mat input_mat, cv::Rect input_rect, int feature_type, cv::Ptr<cv::ml::NormalBayesClassifier> input_classifier, vector<cv::Rect> &output_rects)
{
	vector<float> feature_vec;
	if (feature_type == fhog_type_)
		get_fhog_feature_vector(input_mat, feature_vec);
	else if (feature_type == hog_type_)
		get_hog_feature_vector(input_mat, feature_vec);
	else if (feature_type == lbp_type_)
		get_lbp_feature_vector(input_mat, feature_vec);
	else if (feature_type == glcm_type_)
		get_glcm_feature_vector(input_mat, feature_vec);
	else if (feature_type == fhog_glcm_type_)
		get_fhog_glcm_feature_vector(input_mat, feature_vec);
	int dim = feature_vec.size();
	//cout << dim << endl;
	Mat feature_mat = Mat::zeros(1, dim, CV_32FC1);
	feature_vec2mat(feature_vec, feature_mat);
	feature_vec.clear();
	float response = input_classifier->predict(feature_mat);
	if (response == 1)
	{
		output_rects.push_back(input_rect);
	}
}