//#include "DataAugment.h"
//#include "sx_data_augment.h"
//void main()
//{
//	for (int i=1;i<=100;i++)
//	{
//		DataAugment da;
//
//		cv::Mat src = cv::imread("E:\\Paper_4_3\\src_4_3_gai\\test\\"+ to_string(i) + ".jpg", IMREAD_UNCHANGED);
//		if (src.empty())
//		{
//			continue;
//		}
//		//imshow("src", src);
//		//waitKey();
//
//		string dstpathname = "E:\\Paper_4_3\\src_4_3_gai\\test2\\";
//		imwrite(dstpathname + to_string(i) + ".jpg", src);
//
//		
//		for (int ii = 1; ii <= 21; ii = ii + 2)
//		{
//			Mat tmp;
//			da.imageRotate(src, tmp, ii, AU_ANG);
//
//			string imwritename = dstpathname + to_string(i)+"_r_" + to_string(ii) + ".jpg";
//			//imshow(imwritename, tmp);
//			//waitKey();
//			imwrite(imwritename, tmp);
//		}
//
//		for (int ii = 1; ii <= 21; ii = ii + 2)
//		{
//			Mat t1, t2, t3, t4;
//			translation(src, t1, t2, t3, t4, ii, ii);
//
//			string imwritename1 = dstpathname + to_string(i)+ "_t1_" + to_string(ii) + ".jpg";
//			imwrite(imwritename1, t1);
//
//			string imwritename2 = dstpathname + to_string(i)+"_t2_" + to_string(ii) + ".jpg";
//			imwrite(imwritename2, t2);
//
//			string imwritename3 = dstpathname + to_string(i)+"_t3_" + to_string(ii) + ".jpg";
//			imwrite(imwritename3, t3);
//
//			string imwritename4 = dstpathname + to_string(i)+"_t4_" + to_string(ii) + ".jpg";
//			imwrite(imwritename4, t4);
//		}
//	}
//
//	//waitKey(0);
//}
