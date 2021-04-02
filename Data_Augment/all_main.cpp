//#include "DataAugment.h"
//#include "sx_data_augment.h"
//
//void enhance_and_save(string inputname, Mat input)
//{
//	imwrite(inputname, input);
//
//	vector<Mat> enhanceMats(3);
//	Mat equ, lap, log, gam, adg;
//	//HistogramEqualization(src, enhanceMats[0]);
//	//LaplacianEnhance(src, enhanceMats[1]);
//	LogarithmicTransformation(input, enhanceMats[0]);
//	GammaTransform(input, enhanceMats[1]);
//	AdaptGammaEnhance(input, enhanceMats[2]);
//
//	for (int j = 0; j < enhanceMats.size(); j++)
//	{
//		string newname = inputname + "enhance" + to_string(j) + ".jpg";
//		//imshow(newname, enhanceMats[j]);
//		//waitKey();
//		imwrite(newname, enhanceMats[j]);
//	}
//}
//
//void main()
//{
//	for (int i=1;i<=93;i++)
//	{
//		string name = "E:\\孙鑫论文\\年前终极整理\\数据集\\那93张\\" + to_string(i) +".jpg";
//		cv::Mat src = cv::imread(name, IMREAD_UNCHANGED);
//		if (src.empty()){
//			continue;
//		}
//		imwrite("E:\\mydata3\\" + to_string(i) + ".jpg", src);
//
//		DataAugment da;
//		vector<Mat> rotateMats(5);
//		da.imageRotate(src, rotateMats[0], 10, AU_ANG);
//		da.imageRotate(src, rotateMats[1], 20, AU_ANG);
//		da.imageRotate(src, rotateMats[2], 50, AU_ANG);
//		da.imageRotate(src, rotateMats[3], 70, AU_ANG);
//		da.imageRotate(src, rotateMats[4], 90, AU_ANG);
//		vector<Mat> flipMats(2);
//		da.imageFlip(src, flipMats[0], MIRROR_V);
//		da.imageFlip(src, flipMats[1], MIRROR_H);
//
//		vector<Mat> tranMats(16);
//		translation(src, tranMats[0], tranMats[1], tranMats[2], tranMats[3], 50, 50);
//		translation(src, tranMats[4], tranMats[5], tranMats[6], tranMats[7], 100, 100);
//		translation(src, tranMats[8], tranMats[9], tranMats[10], tranMats[11], 50, 100);
//		translation(src, tranMats[12], tranMats[13], tranMats[14], tranMats[15], 100, 50);
//
//		for (int j = 0; j < rotateMats.size(); j++)
//		{ 
//			string newname = "E:\\mydata3\\" + to_string(i) + "rotate" + to_string(j) + ".jpg";
//			enhance_and_save(newname, rotateMats[j]);
//		}
//
//		for (int j = 0; j < flipMats.size(); j++)
//		{
//			string newname = "E:\\mydata3\\" + to_string(i) + "flip" + to_string(j) + ".jpg";
//			enhance_and_save(newname, flipMats[j]);
//		}
//
//		for (int j = 0; j < tranMats.size(); j++)
//		{
//			string newname = "E:\\mydata3\\" + to_string(i) + "tran" + to_string(j) + ".jpg";
//			enhance_and_save(newname, tranMats[j]);
//		}
//
//
//
//
//
//	}
//
//	//waitKey(0);
//}
