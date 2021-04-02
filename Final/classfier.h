#ifndef CLASSFIER_H_
#define CLASSFIER_H_

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <io.h>
#include "feature.h"

using namespace std;
using namespace cv::ml;
using namespace cv;

namespace IPSG
{
	class Classfier
	{
	public:
		Classfier()
		{}
		~Classfier()
		{}
		void test_and_save_classifier(const Ptr<StatModel>& model, const Mat& data, const Mat& responses,
			int ntrain_samples, int rdelta);
		void test(const Ptr<StatModel>& model, const Mat& data, const Mat& responses,
			int ntrain_samples, int rdelta);
		bool build_rtrees_classifier(const string& data_filename);
		bool build_boost_classifier(const string& data_filename);
		bool build_mlp_classifier(const string& data_filename);
		bool build_knearest_classifier(const string& data_filename, int K);
		bool build_nbayes_classifier(const string& data_filename);
		bool build_svm_classifier(const string& data_filename);
		bool testRF(std::string modelPath, std::string imagePath, int response);
	private:
		inline TermCriteria TC(int iters, double eps);
		bool read_num_class_data(const string& filename, int var_count, Mat* _data, Mat* _responses);
		Ptr<TrainData> prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples);
	};
}
#endif 

