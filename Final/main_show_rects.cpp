#include "gen_features.h"
#include "process_rects.h"
#include "unsupervised.h"
#include "supervised.h"

//cv::Rect tmp;
//mergeRects(fhog_glcm_svm_rects, tmp);
//fhog_glcm_svm_rects.clear();
//fhog_glcm_svm_rects.push_back(tmp);

////bayes+lbp can not finish training.
////rtrees+lbp can not finish training.

int main()
{
	vector<vector<cv::Rect>> groundtruth_rects;
	get_groundtruth_rects(groundtruth_filename, groundtruth_rects);
	vector<Mat> detect_mats;
	load_images(detestTestName, detect_mats, 0);
	if (groundtruth_rects.size() != detect_mats.size())
	{
		cout << "groundtruth_rects.size() != detect_mats.size()" << endl;
		return 0;
	}

	string fhog_svm_filename, hog_svm_filename, lbp_svm_filename, glcm_svm_filename, fhog_glcm_svm_filename;

	//train(PosPath, NegPath, fhog_type_, svm_type_, fhog_svm_filename);
	//cv::Ptr<cv::ml::SVM> svm_fhog = cv::ml::SVM::load(fhog_svm_filename);
	//train(PosPath, NegPath, hog_type_, svm_type_, hog_svm_filename);
	//cv::Ptr<cv::ml::SVM> svm_hog = cv::ml::SVM::load(hog_svm_filename);
	////train(PosPath, NegPath, lbp_type_, svm_type_, lbp_svm_filename);
	////cv::Ptr<cv::ml::SVM> svm_lbp = cv::ml::SVM::load(lbp_svm_filename);
	//train(PosPath, NegPath, glcm_type_, svm_type_, glcm_svm_filename);
	//cv::Ptr<cv::ml::SVM> svm_glcm = cv::ml::SVM::load(glcm_svm_filename);
	train(PosPath, NegPath, fhog_glcm_type_, svm_type_, fhog_glcm_svm_filename);
	cv::Ptr<cv::ml::SVM> svm_fhog_glcm = cv::ml::SVM::load(fhog_glcm_svm_filename);

	//string fhog_knn_filename, hog_knn_filename, lbp_knn_filename, glcm_knn_filename, fhog_glcm_knn_filename;

	//train(PosPath, NegPath, fhog_type_, knn_type_, fhog_knn_filename);
	//cv::Ptr<cv::ml::KNearest> knn_fhog = cv::Algorithm::load<cv::ml::KNearest>(fhog_knn_filename);
	//train(PosPath, NegPath, hog_type_, knn_type_, hog_knn_filename);
	//cv::Ptr<cv::ml::KNearest> knn_hog = cv::Algorithm::load<cv::ml::KNearest>(hog_knn_filename);
	//train(PosPath, NegPath, lbp_type_, knn_type_, lbp_knn_filename);
	//cv::Ptr<cv::ml::KNearest> knn_lbp = cv::Algorithm::load<cv::ml::KNearest>(lbp_knn_filename);
	//train(PosPath, NegPath, glcm_type_, knn_type_, glcm_knn_filename);
	//cv::Ptr<cv::ml::KNearest> knn_glcm = cv::Algorithm::load<cv::ml::KNearest>(glcm_knn_filename);
	//train(PosPath, NegPath, fhog_glcm_type_, knn_type_, fhog_glcm_knn_filename);
	//cv::Ptr<cv::ml::KNearest> knn_fhog_glcm = cv::Algorithm::load<cv::ml::KNearest>(fhog_glcm_knn_filename);

	//string fhog_bayes_filename, hog_bayes_filename, lbp_bayes_filename, glcm_bayes_filename, fhog_glcm_bayes_filename;

	//train(PosPath, NegPath, fhog_type_, bayes_type_, fhog_bayes_filename);
	//cv::Ptr<cv::ml::NormalBayesClassifier> bayes_fhog = cv::Algorithm::load<cv::ml::NormalBayesClassifier>(fhog_bayes_filename);
	//train(PosPath, NegPath, hog_type_, bayes_type_, hog_bayes_filename);
	//cv::Ptr<cv::ml::NormalBayesClassifier> bayes_hog = cv::Algorithm::load<cv::ml::NormalBayesClassifier>(hog_bayes_filename);
	//train(PosPath, NegPath, lbp_type_, bayes_type_, lbp_bayes_filename);
	//cv::Ptr<cv::ml::NormalBayesClassifier> bayes_lbp = cv::Algorithm::load<cv::ml::NormalBayesClassifier>(lbp_bayes_filename);
	//train(PosPath, NegPath, glcm_type_, bayes_type_, glcm_bayes_filename);
	//cv::Ptr<cv::ml::NormalBayesClassifier> bayes_glcm = cv::Algorithm::load<cv::ml::NormalBayesClassifier>(glcm_bayes_filename);
	//train(PosPath, NegPath, fhog_glcm_type_, bayes_type_, fhog_glcm_bayes_filename);
	//cv::Ptr<cv::ml::NormalBayesClassifier> bayes_fhog_glcm = cv::Algorithm::load<cv::ml::NormalBayesClassifier>(fhog_glcm_bayes_filename);

	//string fhog_rtrees_filename, hog_rtrees_filename, lbp_rtrees_filename, glcm_rtrees_filename, fhog_glcm_rtrees_filename;
	//train(PosPath, NegPath, fhog_type_, rtrees_type_, fhog_rtrees_filename);
	//cv::Ptr<cv::ml::RTrees> rtrees_fhog = cv::Algorithm::load<cv::ml::RTrees>(fhog_rtrees_filename);
	//train(PosPath, NegPath, hog_type_, rtrees_type_, hog_rtrees_filename);
	//cv::Ptr<cv::ml::RTrees> rtrees_hog = cv::Algorithm::load<cv::ml::RTrees>(hog_rtrees_filename);
	//train(PosPath, NegPath, lbp_type_, rtrees_type_, lbp_rtrees_filename);
	//cv::Ptr<cv::ml::RTrees> rtrees_lbp = cv::Algorithm::load<cv::ml::RTrees>(lbp_rtrees_filename);
	//train(PosPath, NegPath, glcm_type_, rtrees_type_, glcm_rtrees_filename);
	//cv::Ptr<cv::ml::RTrees> rtrees_glcm = cv::Algorithm::load<cv::ml::RTrees>(glcm_rtrees_filename);
	//train(PosPath, NegPath, fhog_glcm_type_, rtrees_type_, fhog_glcm_rtrees_filename);
	//cv::Ptr<cv::ml::RTrees> rtrees_fhog_glcm = cv::Algorithm::load<cv::ml::RTrees>(fhog_glcm_rtrees_filename);

	//string fhog_adaboost_filename, hog_adaboost_filename, lbp_adaboost_filename, glcm_adaboost_filename, fhog_glcm_adaboost_filename;

	//train(PosPath, NegPath, fhog_type_, adaboost_type_, fhog_adaboost_filename);
	//cv::Ptr<cv::ml::Boost> adaboost_fhog = cv::Algorithm::load<cv::ml::Boost>(fhog_adaboost_filename);
	//train(PosPath, NegPath, hog_type_, adaboost_type_, hog_adaboost_filename);
	//cv::Ptr<cv::ml::Boost> adaboost_hog = cv::Algorithm::load<cv::ml::Boost>(hog_adaboost_filename);
	//train(PosPath, NegPath, lbp_type_, adaboost_type_, lbp_adaboost_filename);
	//cv::Ptr<cv::ml::Boost> adaboost_lbp = cv::Algorithm::load<cv::ml::Boost>(lbp_adaboost_filename);
	//train(PosPath, NegPath, glcm_type_, adaboost_type_, glcm_adaboost_filename);
	//cv::Ptr<cv::ml::Boost> adaboost_glcm = cv::Algorithm::load<cv::ml::Boost>(glcm_adaboost_filename);
	//train(PosPath, NegPath, fhog_glcm_type_, adaboost_type_, fhog_glcm_adaboost_filename);
	//cv::Ptr<cv::ml::Boost> adaboost_fhog_glcm = cv::Algorithm::load<cv::ml::Boost>(fhog_glcm_adaboost_filename);

	//string fhog_ann_filename, hog_ann_filename, lbp_ann_filename, glcm_ann_filename, fhog_glcm_ann_filename;

	//train(PosPath, NegPath, fhog_type_, ann_type_, fhog_ann_filename);
	//cv::Ptr<cv::ml::ANN_MLP> ann_fhog = cv::Algorithm::load<cv::ml::ANN_MLP>(fhog_ann_filename);
	//train(PosPath, NegPath, hog_type_, ann_type_, hog_ann_filename);
	//cv::Ptr<cv::ml::ANN_MLP> ann_hog = cv::Algorithm::load<cv::ml::ANN_MLP>(hog_ann_filename);
	//train(PosPath, NegPath, lbp_type_, ann_type_, lbp_ann_filename);
	//cv::Ptr<cv::ml::ANN_MLP> ann_lbp = cv::Algorithm::load<cv::ml::ANN_MLP>(lbp_ann_filename);
	//train(PosPath, NegPath, glcm_type_, ann_type_, glcm_ann_filename);
	//cv::Ptr<cv::ml::ANN_MLP> ann_glcm = cv::Algorithm::load<cv::ml::ANN_MLP>(glcm_ann_filename);
	//train(PosPath, NegPath, fhog_glcm_type_, ann_type_, fhog_glcm_ann_filename);
	//cv::Ptr<cv::ml::ANN_MLP> ann_fhog_glcm = cv::Algorithm::load<cv::ml::ANN_MLP>(fhog_glcm_ann_filename);

	for (size_t i = 0; i < detect_mats.size(); i++)
	{
		vector<cv::Rect> fhog_glcm_svm_rects;
		////vector<cv::Rect> lbp_svm_rects;
		//vector<cv::Rect> glcm_svm_rects;
		//vector<cv::Rect> fhog_svm_rects;
		//vector<cv::Rect> hog_svm_rects;

		//vector<cv::Rect> fhog_glcm_knn_rects;
		//vector<cv::Rect> lbp_knn_rects;
		//vector<cv::Rect> glcm_knn_rects;
		//vector<cv::Rect> fhog_knn_rects;
		//vector<cv::Rect> hog_knn_rects;

		//vector<cv::Rect> fhog_glcm_bayes_rects;
		//vector<cv::Rect> lbp_bayes_rects;
		//vector<cv::Rect> glcm_bayes_rects;
		//vector<cv::Rect> fhog_bayes_rects;
		//vector<cv::Rect> hog_bayes_rects;

		//vector<cv::Rect> fhog_glcm_rtrees_rects;
		//vector<cv::Rect> lbp_rtrees_rects;
		//vector<cv::Rect> glcm_rtrees_rects;
		//vector<cv::Rect> fhog_rtrees_rects;
		//vector<cv::Rect> hog_rtrees_rects;

		//vector<cv::Rect> fhog_glcm_adaboost_rects;
		//vector<cv::Rect> lbp_adaboost_rects;
		//vector<cv::Rect> glcm_adaboost_rects;
		//vector<cv::Rect> fhog_adaboost_rects;
		//vector<cv::Rect> hog_adaboost_rects;

		//vector<cv::Rect> fhog_glcm_ann_rects;
		//vector<cv::Rect> lbp_ann_rects;
		//vector<cv::Rect> glcm_ann_rects;
		//vector<cv::Rect> fhog_ann_rects;
		//vector<cv::Rect> hog_ann_rects;

		double start = static_cast<double>(cv::getTickCount());
		for (int y = 0; y < detect_mats[i].cols - winsize + 1; y = y + stride)
		{
			for (int x = 0; x < detect_mats[i].rows - winsize + 1; x = x + stride)
			{
				cv::Rect rect_tmp(x, y, winsize, winsize);
				Mat roi = detect_mats[i](rect_tmp).clone();


				////Mat myphase2 = cv::imread("278.jpg");
				////cv::imshow("ss",myphase2);
				////cv::waitKey();

				svm_classify(roi, rect_tmp, fhog_glcm_type_, svm_fhog_glcm, fhog_glcm_svm_rects);
				show_rects("fhog_glcm_svm_rects", detect_mats[i], fhog_glcm_svm_rects);
				//////////
				//svm_classify(roi, rect_tmp, lbp_type_, svm_lbp, lbp_svm_rects);
				//show_rects("lbp_svm_rects", detect_mats[i], lbp_svm_rects);
				//////////
				//svm_classify(roi, rect_tmp, glcm_type_, svm_glcm, glcm_svm_rects);
				//show_rects("glcm_svm_rects", detect_mats[i], glcm_svm_rects);
				////////////
				//svm_classify(roi, rect_tmp, fhog_type_, svm_fhog, fhog_svm_rects);
				//show_rects("fhog_svm_rects", detect_mats[i], fhog_svm_rects);
				///////////
				//svm_classify(roi, rect_tmp, hog_type_, svm_hog, hog_svm_rects);
				//show_rects("hog_svm_rects", detect_mats[i], hog_svm_rects);
			
				//knn_classify(roi, rect_tmp, fhog_glcm_type_, knn_fhog_glcm, fhog_glcm_knn_rects);
				//show_rects("fhog_glcm_knn_rects", detect_mats[i], fhog_glcm_knn_rects);
				/////////
				//knn_classify(roi, rect_tmp, lbp_type_, knn_lbp, lbp_knn_rects);
				//show_rects("lbp_knn_rects", detect_mats[i], lbp_knn_rects);
				//////////
				//knn_classify(roi, rect_tmp, glcm_type_, knn_glcm, glcm_knn_rects);
				//show_rects("glcm_knn_rects", detect_mats[i], glcm_knn_rects);
				//////////
				//knn_classify(roi, rect_tmp, fhog_type_, knn_fhog, fhog_knn_rects);
				//show_rects("fhog_knn_rects", detect_mats[i], fhog_knn_rects);
				////////////
				//knn_classify(roi, rect_tmp, hog_type_, knn_hog, hog_knn_rects);
				//show_rects("hog_knn_rects", detect_mats[i], hog_knn_rects);

				//bayes_classify(roi, rect_tmp, fhog_glcm_type_, bayes_fhog_glcm, fhog_glcm_bayes_rects);
				//show_rects("fhog_glcm_bayes_rects", detect_mats[i], fhog_glcm_bayes_rects);
				//////////
				//bayes_classify(roi, rect_tmp, lbp_type_, bayes_lbp, lbp_bayes_rects);
				//show_rects("lbp_bayes_rects", detect_mats[i], lbp_bayes_rects);
				///////////
				//bayes_classify(roi, rect_tmp, glcm_type_, bayes_glcm, glcm_bayes_rects);
				//show_rects("glcm_bayes_rects", detect_mats[i], glcm_bayes_rects);
				//////////
				//bayes_classify(roi, rect_tmp, fhog_type_, bayes_fhog, fhog_bayes_rects);
				//show_rects("fhog_bayes_rects", detect_mats[i], fhog_bayes_rects);
				//////////
				//bayes_classify(roi, rect_tmp, hog_type_, bayes_hog, hog_bayes_rects);
				//show_rects("hog_bayes_rects", detect_mats[i], hog_bayes_rects);
			

				//rtrees_classify(roi, rect_tmp, fhog_glcm_type_, rtrees_fhog_glcm, fhog_glcm_rtrees_rects);
				//show_rects("fhog_glcm_rtrees_rects", detect_mats[i], fhog_glcm_rtrees_rects);
				//////////
				//rtrees_classify(roi, rect_tmp, lbp_type_, rtrees_lbp, lbp_rtrees_rects);
				//show_rects("lbp_rtrees_rects", detect_mats[i], lbp_rtrees_rects);
				///////////
				//rtrees_classify(roi, rect_tmp, glcm_type_, rtrees_glcm, glcm_rtrees_rects);
				//show_rects("glcm_rtrees_rects", detect_mats[i], glcm_rtrees_rects);
				//////////
				//rtrees_classify(roi, rect_tmp, fhog_type_, rtrees_fhog, fhog_rtrees_rects);
				//show_rects("fhog_rtrees_rects", detect_mats[i], fhog_rtrees_rects);
				//////////
				//rtrees_classify(roi, rect_tmp, hog_type_, rtrees_hog, hog_rtrees_rects);
				//show_rects("hog_rtrees_rects", detect_mats[i], hog_rtrees_rects);
			
				//adaboost_classify(roi, rect_tmp, fhog_glcm_type_, adaboost_fhog_glcm, fhog_glcm_adaboost_rects);
				//show_rects("fhog_glcm_adaboost_rects", detect_mats[i], fhog_glcm_adaboost_rects);
				//////////
				//adaboost_classify(roi, rect_tmp, lbp_type_, adaboost_lbp, lbp_adaboost_rects);
				//show_rects("lbp_adaboost_rects", detect_mats[i], lbp_adaboost_rects);
				///////////
				//adaboost_classify(roi, rect_tmp, glcm_type_, adaboost_glcm, glcm_adaboost_rects);
				//show_rects("glcm_adaboost_rects", detect_mats[i], glcm_adaboost_rects);
				//////////
				//adaboost_classify(roi, rect_tmp, fhog_type_, adaboost_fhog, fhog_adaboost_rects);
				//show_rects("fhog_adaboost_rects", detect_mats[i], fhog_adaboost_rects);
				//////////
				//adaboost_classify(roi, rect_tmp, hog_type_, adaboost_hog, hog_adaboost_rects);
				//show_rects("hog_adaboost_rects", detect_mats[i], hog_adaboost_rects);

				//ann_classify(roi, rect_tmp, fhog_glcm_type_, ann_fhog_glcm, fhog_glcm_ann_rects);
				//show_rects("fhog_glcm_ann_rects", detect_mats[i], fhog_glcm_ann_rects);
				//////////
				//ann_classify(roi, rect_tmp, lbp_type_, ann_lbp, lbp_ann_rects);
				//show_rects("lbp_ann_rects", detect_mats[i], lbp_ann_rects);
				///////////
				//ann_classify(roi, rect_tmp, glcm_type_, ann_glcm, glcm_ann_rects);
				//show_rects("glcm_ann_rects", detect_mats[i], glcm_ann_rects);
				//////////
				//ann_classify(roi, rect_tmp, fhog_type_, ann_fhog, fhog_ann_rects);
				//show_rects("fhog_ann_rects", detect_mats[i], fhog_ann_rects);
				//////////
				//ann_classify(roi, rect_tmp, hog_type_, ann_hog, hog_ann_rects);
				//show_rects("hog_ann_rects", detect_mats[i], hog_ann_rects);
			
			}
		}
		double time = ((double)cv::getTickCount() - start) / cv::getTickFrequency();
		cout << "ml method time is :" << time << endl;










		double start2 = static_cast<double>(cv::getTickCount());

		//Mat bms = detect_mats[i].clone();
		//vector<cv::Rect> bms_rects;
		//get_saliency_rects(bms, bms_type_, bms_rects);
		//show_rects("bms", bms, bms_rects);


		//Mat hc = detect_mats[i].clone();
		//vector<cv::Rect> hc_rects;
		//get_saliency_rects(hc, hc_type_, hc_rects);
		//show_rects("hc", hc, hc_rects);

		//Mat sr = detect_mats[i].clone();
		//vector<cv::Rect> sr_rects;
		//get_saliency_rects(sr, sr_type_, sr_rects);
		//show_rects("sr", sr, sr_rects);


		//Mat uav = detect_mats[i].clone();
		//vector<cv::Rect> uav_rects;
		//get_saliency_rects(uav, uav_type_, uav_rects);
		//show_rects("uav", uav, uav_rects);


		Mat myphase2 = cv::imread("278.jpg");
		//cv::imshow("ss",myphase2);
		//cv::waitKey();
		//Mat myphase2 = detect_mats[i].clone();
		vector<cv::Rect> myphase2_rects;
		get_saliency_rects(myphase2, myphase2_type_, myphase2_rects);
		show_rects("myphase2", myphase2, myphase2_rects);
		

		double time2 = ((double)cv::getTickCount() - start) / cv::getTickFrequency();
		cout << "saliency method time is :" << time2 << endl;


		///////////////////////////////////////mymethod
		//Mat mat_mymethod = detect_mats[i].clone();
		//vector<cv::Rect> mat_mymethod_rects;
		//combine_two_methods_rects(fhog_glcm_svm_rects, myphase2_rects, mat_mymethod_rects, 0.1, 0.7);
		//cv::Rect paper_mymethod_rect;
		//mergeRects(mat_mymethod_rects, paper_mymethod_rect);
		//mat_mymethod_rects.clear();
		//mat_mymethod_rects.push_back(paper_mymethod_rect);
		//show_rects("mat_mymethod_rects", mat_mymethod, mat_mymethod_rects);


		///////////////////////////////////////calculate my method rects
		//Mat mat_mymethod_correct = detect_mats[i].clone();
		//vector<cv::Rect> mymethod_correct_rects;
		//get_correct_rects(mat_mymethod_rects, groundtruth_rects[i], 0.5, mymethod_correct_rects);
		//show_rects("mat_mymethod_correct_rects", mat_mymethod_correct, mymethod_correct_rects);




		cv::waitKey(0);
	}

	return 0;
}
