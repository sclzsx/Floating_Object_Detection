#include "classfier.h"

bool IPSG::Classfier::read_num_class_data(const string& filename, int var_count, Mat* _data, Mat* _responses) //var_count表示读取特征的维数
{
	const int M = 1024 * 2;
	char buf[M + 2];

	Mat el_ptr(1, var_count, CV_32F);
	int i;
	vector<int> responses;

	_data->release();
	_responses->release();
	FILE *f;
	fopen_s(&f, filename.c_str(), "rt");
	if (!f)
	{
		cout << "Could not read the database " << filename << endl;
		return false;
	}

	for (;;)
	{
		char* ptr;
		if (!fgets(buf, M, f) || !strchr(buf, ' '))  //fgets从文件结构体指针stream中读取数据保存在buf中，每次读取一行，每次存储M大小
			break;    //strchar查找字符串buf中首次出现","的位置,并返回首次出现的指针，表明数据存储的格式是以","为间隔
		responses.push_back((int)buf[0]);  //数据存储第零位是响应
		ptr = buf + 2;  //数据存储第二位是特征
		for (i = 0; i < var_count; i++)
		{
			int n = 0;
			sscanf_s(ptr, "%f%n", &el_ptr.at<float>(i), &n);  //
			ptr += n + 1;
		}
		if (i < var_count)
			break;
		_data->push_back(el_ptr);
	}
	fclose(f);
	Mat(responses).copyTo(*_responses);
	return true;
}

Ptr<TrainData> IPSG::Classfier::prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples)
{
	Mat sample_idx = Mat::zeros(1, data.rows, CV_8U);
	Mat train_samples = sample_idx.colRange(0, ntrain_samples);  //取特定的行，0~ntrain_samples
	train_samples.setTo(Scalar::all(1));

	int nvars = data.cols;
	Mat var_type(nvars + 1, 1, CV_8U);
	var_type.setTo(Scalar::all(VAR_ORDERED));
	var_type.at<uchar>(nvars) = VAR_CATEGORICAL;

	return TrainData::create(data, ROW_SAMPLE, responses,
		noArray(), sample_idx, noArray(), var_type);
}

inline TermCriteria IPSG::Classfier::TC(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}
void IPSG::Classfier::test(const Ptr<StatModel>& model, const Mat& data, const Mat& responses,
	int ntrain_samples, int rdelta)
{
	int i, nsamples_all = data.rows;
	double train_hr = 0, test_hr = 0;

	// compute prediction error on train and test data
	for (i = 0; i < nsamples_all; i++)
	{
		Mat sample = data.row(i);

		float r = model->predict(sample);
		r = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

		if (i < ntrain_samples)
			train_hr += r;
		else
			test_hr += r;
	}

	test_hr /= nsamples_all - ntrain_samples;
	train_hr = ntrain_samples > 0 ? train_hr / ntrain_samples : 1.;

	printf("Recognition rate: train = %.1f%%, test = %.1f%%\n",
		train_hr*100., test_hr*100.);
}

void IPSG::Classfier::test_and_save_classifier(const Ptr<StatModel>& model, const Mat& data, const Mat& responses,
	int ntrain_samples, int rdelta)
{
	int i, nsamples_all = data.rows;
	double train_hr = 0, test_hr = 0;
	double Ptrain_hr = 0, Ntrain_hr = 0, Ptest_hr = 0, Ntest_hr = 0;
	double PtrainSum = 0, PtestSum = 0, NtrainSum = 0, NtestSum = 0;
	for (int i = 0; i < responses.rows; i++)
	{
		if (i < ntrain_samples && responses.at<int>(i) == 49)
		{
			PtrainSum++;
		}
		if (i < ntrain_samples && responses.at<int>(i) == 45)
		{
			NtrainSum++;
		}
		if (i >= ntrain_samples && responses.at<int>(i) == 49)
		{
			PtestSum++;
		}
		if (i >= ntrain_samples && responses.at<int>(i) == 45)
		{
			NtestSum++;
		}
	}
	// compute prediction error on train and test data

	test_hr /= nsamples_all - ntrain_samples;
	train_hr = ntrain_samples > 0 ? train_hr / ntrain_samples : 1.;

	printf("Recognition rate: train = %.1f%%, test = %.1f%%\n",
		train_hr*100., test_hr*100.);
}

bool IPSG::Classfier::build_rtrees_classifier(const string& data_filename)
{
	Mat data;
	Mat responses;
	read_num_class_data(data_filename, 148, &data, &responses);

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.5);

	Ptr<RTrees> model;
	Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
	model = RTrees::create();
	model->setMaxDepth(21);
	model->setMinSampleCount(21);
	model->setRegressionAccuracy(0);
	model->setUseSurrogates(false);
	model->setMaxCategories(2);
	model->setPriors(Mat());
	model->setCalculateVarImportance(true);
	model->setActiveVarCount(4);
	model->setTermCriteria(TC(100, 0.01f));
	model->train(tdata);
	test_and_save_classifier(model, data, responses, ntrain_samples, 0);
	cout << "Number of trees: " << model->getRoots().size() << endl;

	// Print variable importance
	Mat var_importance = model->getVarImportance();
	if (!var_importance.empty())
	{
		double rt_imp_sum = sum(var_importance)[0];
		printf("var#\timportance (in %%):\n");
		int i, n = (int)var_importance.total();
		for (i = 0; i < n; i++)
			printf("%-2d\t%-4.1f\n", i, 100.f*var_importance.at<float>(i) / rt_imp_sum);
	}

	return true;
}

bool IPSG::Classfier::build_boost_classifier(const string& data_filename)
{
	const int class_count = 26;
	Mat data;
	Mat responses;
	Mat weak_responses;

	read_num_class_data(data_filename, 16, &data, &responses);
	int i, j, k;
	Ptr<Boost> model;

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.5);
	int var_count = data.cols;

	Mat new_data(ntrain_samples*class_count, var_count + 1, CV_32F);
	Mat new_responses(ntrain_samples*class_count, 1, CV_32S);

	for (i = 0; i < ntrain_samples; i++)
	{
		const float* data_row = data.ptr<float>(i);
		for (j = 0; j < class_count; j++)
		{
			float* new_data_row = (float*)new_data.ptr<float>(i*class_count + j);
			memcpy(new_data_row, data_row, var_count * sizeof(data_row[0]));
			new_data_row[var_count] = (float)j;
			new_responses.at<int>(i*class_count + j) = responses.at<int>(i) == j + 'A';
		}
	}

	Mat var_type(1, var_count + 2, CV_8U);
	var_type.setTo(Scalar::all(VAR_ORDERED));
	var_type.at<uchar>(var_count) = var_type.at<uchar>(var_count + 1) = VAR_CATEGORICAL;

	Ptr<TrainData> tdata = TrainData::create(new_data, ROW_SAMPLE, new_responses,
		noArray(), noArray(), noArray(), var_type);
	vector<double> priors(2);
	priors[0] = 1;
	priors[1] = 26;

	model = Boost::create();
	model->setBoostType(Boost::GENTLE);
	model->setWeakCount(50);
	model->setWeightTrimRate(0.95);
	model->setMaxDepth(5);
	model->setUseSurrogates(false);
	model->setPriors(Mat(priors));
	model->train(tdata);
	Mat temp_sample(1, var_count + 1, CV_32F);
	float* tptr = temp_sample.ptr<float>();

	// compute prediction error on train and test data
	double train_hr = 0, test_hr = 0;
	for (i = 0; i < nsamples_all; i++)
	{
		int best_class = 0;
		double max_sum = -DBL_MAX;
		const float* ptr = data.ptr<float>(i);
		for (k = 0; k < var_count; k++)
			tptr[k] = ptr[k];

		for (j = 0; j < class_count; j++)
		{
			tptr[var_count] = (float)j;
			float s = model->predict(temp_sample, noArray(), StatModel::RAW_OUTPUT);
			if (max_sum < s)
			{
				max_sum = s;
				best_class = j + 'A';
			}
		}

		double r = std::abs(best_class - responses.at<int>(i)) < FLT_EPSILON ? 1 : 0;
		if (i < ntrain_samples)
			train_hr += r;
		else
			test_hr += r;
	}

	test_hr /= nsamples_all - ntrain_samples;
	train_hr = ntrain_samples > 0 ? train_hr / ntrain_samples : 1.;
	printf("Recognition rate: train = %.1f%%, test = %.1f%%\n",
		train_hr*100., test_hr*100.);

	cout << "Number of trees: " << model->getRoots().size() << endl;
	return true;
}

bool IPSG::Classfier::build_mlp_classifier(const string& data_filename)
{
	const int class_count = 2;
	Mat data;
	Mat responses;

	read_num_class_data(data_filename, 16, &data, &responses);
	Ptr<ANN_MLP> model;

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.8);
	Mat train_data = data.rowRange(0, ntrain_samples);
	Mat train_responses = Mat::zeros(ntrain_samples, class_count, CV_32F);

	// 1. unroll the responses
	cout << "Unrolling the responses...\n";
	for (int i = 0; i < ntrain_samples; i++)
	{
		int cls_label = responses.at<int>(i) - 'A';
		train_responses.at<float>(i, cls_label) = 1.f;
	}

	// 2. train classifier
	int layer_sz[] = { data.cols, 100, 100, class_count };
	int nlayers = (int)(sizeof(layer_sz) / sizeof(layer_sz[0]));
	Mat layer_sizes(1, nlayers, CV_32S, layer_sz);

#if 1
	int method = ANN_MLP::BACKPROP;
	double method_param = 0.001;
	int max_iter = 300;
#else
	int method = ANN_MLP::RPROP;
	double method_param = 0.1;
	int max_iter = 1000;
#endif

	Ptr<TrainData> tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses);
	model = ANN_MLP::create();
	model->setLayerSizes(layer_sizes);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
	model->setTermCriteria(TC(max_iter, 0));
	model->setTrainMethod(method, method_param);
	model->train(tdata);
//	model->save("mlp.xml");
	return true;
}

bool IPSG::Classfier::build_knearest_classifier(const string& data_filename, int K)
{
	Mat data;
	Mat responses;
	read_num_class_data(data_filename, 148, &data, &responses);
	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.8);

	Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
	Ptr<KNearest> model = KNearest::create();
	model->setDefaultK(K);
	model->setIsClassifier(true);
	model->train(tdata);
//	model->save("knn.xml");

	test(model, data, responses, ntrain_samples, 0);
//	analysisOfMissingDetectionRate(model, "F:\\项目\\磁粉\\牛哥\\test\\BAD", 45, "F:\\项目\\磁粉\\MissingDetectionRate\\牛哥\\KNN\\N\\");
	return true;
}

bool IPSG::Classfier::build_nbayes_classifier(const string& data_filename)
{
	Mat data;
	Mat responses;
	read_num_class_data(data_filename, 16, &data, &responses);

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.8);

	Ptr<NormalBayesClassifier> model;
	Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
	model = NormalBayesClassifier::create();
	model->train(tdata);
	model->save("bayes.xml");
	test(model, data, responses, ntrain_samples, 0);
	return true;
}

bool IPSG::Classfier::build_svm_classifier(const string& data_filename)
{
	Mat data;
	Mat responses;
	read_num_class_data(data_filename, 16, &data, &responses);

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.8);

	Ptr<SVM> model;
	Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
	model = SVM::create();
	model->setType(SVM::C_SVC);
	model->setKernel(SVM::LINEAR);
	model->setC(1);
	model->train(tdata);
	model->save("svm.xml");

	test(model, data, responses, ntrain_samples, 0);
	return true;
}
