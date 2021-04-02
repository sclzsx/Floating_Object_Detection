//#include <dlib/svm_threaded.h>
//#include <dlib/gui_widgets.h>
//#include <dlib/image_processing.h>
//#include <dlib/data_io.h>
//#include <dlib/opencv.h>
//#include <iostream>
//#include <fstream>
//#include <opencv2/opencv.hpp>
//
//using namespace std;
//using namespace dlib;
//
//static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r)
//{
//	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
//}
//
//bool rectA_intersect_rectB(cv::Rect rectA, cv::Rect rectB)
//{
//	if (rectA.x > rectB.x + rectB.width) { return false; }
//	if (rectA.y > rectB.y + rectB.height) { return false; }
//	if ((rectA.x + rectA.width) < rectB.x) { return false; }
//	if ((rectA.y + rectA.height) < rectB.y) { return false; }
//
//	float colInt = min(rectA.x + rectA.width, rectB.x + rectB.width) - max(rectA.x, rectB.x);
//	float rowInt = min(rectA.y + rectA.height, rectB.y + rectB.height) - max(rectA.y, rectB.y);
//	float intersection = colInt * rowInt;
//	float areaA = rectA.width * rectA.height;
//	float areaB = rectB.width * rectB.height;
//	float intersectionPercent = intersection / (areaA + areaB - intersection);
//
//	if ((0 < intersectionPercent) && (intersectionPercent < 1) && (intersection != areaA) && (intersection != areaB))
//	{
//		return true;
//	}
//
//	return false;
//}
//
//int flag = 5;
//
//int main(int argc, char** argv)
//{
//	dlib::array<array2d<unsigned char> > images_train, images_test;
//	std::vector<std::vector<rectangle> > obj_boxes_train, obj_boxes_test;
//	load_image_dataset(images_train, obj_boxes_train, "E:\\DataSets\\litter\\src_V2\\src_train.xml");
//	load_image_dataset(images_test, obj_boxes_test, "E:\\DataSets\\litter\\src_V2\\src_test.xml");
//
//	if (flag == 0)
//	{
//		try
//		{
//			//upsample_image_dataset<pyramid_down<2> >(images_train, obj_boxes_train);
//			//upsample_image_dataset<pyramid_down<2> >(images_test, obj_boxes_test);
//			//add_image_left_right_flips(images_train, obj_boxes_train);
//			typedef scan_fhog_pyramid<pyramid_down<1> > image_scanner_type;
//			image_scanner_type scanner;
//			//2. 设置scanner扫描窗口大小
//			scanner.set_detection_window_size(80, 80);
//			//3.定义trainer类型（SVM），用于训练人脸检测器                
//			structural_object_detection_trainer<image_scanner_type> trainer(scanner);
//			// Set this to the number of processing cores on your machine.
//			trainer.set_num_threads(1);
//			// 设置SVM的参数C，C越大表示更好地去拟合训练集，当然也有可能造成过拟合。通过尝试不同C在测试集上的效果得到最佳值
//			trainer.set_c(4);
//			trainer.be_verbose();
//			//设置训练结束条件，"risk gap"<0.01时训练结束，值越小表示SVM优化问题越精确，训练时间也会越久。
//			//通常取0.1-0.01.在verbose模式下每一轮的risk gap都会打印出来。
//			trainer.set_epsilon(0.01);
//			//4.训练，生成object_detector
//			object_detector<image_scanner_type> detector = trainer.train(images_train, obj_boxes_train);
//			// 输出precision, recall, average precision.
//			cout << "training results: " << test_object_detection_function(detector, images_train, obj_boxes_train) << endl;
//			cout << "testing results:  " << test_object_detection_function(detector, images_test, obj_boxes_test) << endl;
//			//image_window hogwin(draw_fhog(detector), "Learned fHOG detector");
//			image_window win;
//			for (unsigned long i = 0; i < images_test.size(); ++i)
//			{
//				std::vector<rectangle> dets = detector(images_test[i]);
//				win.clear_overlay();
//				win.set_image(images_test[i]);
//				win.add_overlay(dets, rgb_pixel(255, 0, 0));
//				cout << "Hit enter to process the next image..." << endl;
//				cin.get();
//			}
//			//四、模型存储
//			serialize("litter_src_V2_80x80_dlibfhogsvm.svm") << detector;
//		}
//		catch (exception& e)
//		{
//			cout << "\nexception thrown!" << endl;
//			cout << e.what() << endl;
//		}
//	}
//	else if(flag == 1)
//	{
//		typedef scan_fhog_pyramid<pyramid_down<1> > image_scanner_type;
//		object_detector<image_scanner_type> detector2;
//		deserialize("litter_src_80x80_dlibfhogsvm.svm") >> detector2;
//		image_window win;
//		for (unsigned long i = 0; i < images_test.size(); ++i)
//		{
//			std::vector<rectangle> dets = detector2(images_test[i]);
//			win.clear_overlay();
//			win.set_image(images_test[i]);
//			win.add_overlay(dets, rgb_pixel(0, 255, 0));
//			cout << "Hit enter to process the next image..." << endl;
//			cin.get();
//		}
//	}
//	else if (flag == 2)
//	{
//		typedef scan_fhog_pyramid<pyramid_down<1> > image_scanner_type;
//		object_detector<image_scanner_type> detector2;
//		deserialize("litter_src_80x80_dlibfhogsvm.svm") >> detector2;
//		image_window win;
//
//		for (int n = 1; n <= 100; n++)
//		{
//			string name = "E:\\lightchange\\test\\" + to_string(n) + ".jpg";
//			cv::Mat src = cv::imread(name);
//			if (src.empty())
//				continue;
//			//cv::Mat gray;
//			//cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);
//
//			dlib::cv_image<rgb_pixel> dlib_img(src);
//			std::vector<rectangle> dets = detector2(dlib_img);
//			win.clear_overlay();
//			//win.set_image(dlib_img);
//			//win.add_overlay(dets, rgb_pixel(0, 255, 0));
//			
//			std::vector<cv::Rect> rects;
//			for (int j = 0; j < dets.size(); j++)
//			{
//				cv::Rect rectTmp = dlibRectangleToOpenCV(dets[j]);
//				cv::Rect rectTmpNew(rectTmp.x+1,rectTmp.y+1,rectTmp.width,rectTmp.height);
//				rects.push_back(rectTmpNew);
//			}
//			for (int i = 0; i < rects.size(); i++)
//			{
//				cv::rectangle(src,rects[i],cv::Scalar(0,0,255));
//			}
//			cv::imshow(name,src);
//			cv::waitKey();
//			cv::imwrite(name,src);
//			//string na = to_string(n) + ".png";
//			//save_png(dlib_img, na);
//			//cout << "Hit enter to process the next image..." << endl;
//			//cin.get();
//		}
//	}
//	else if (flag == 3)//在数据集中裁剪出64x64的正负样本
//	{
//		int train_num = obj_boxes_train.size();
//		int test_num = obj_boxes_test.size();
//
//		for (int i=0; i < train_num; i++)
//		{
//			cv::Rect rectTmp = dlibRectangleToOpenCV(obj_boxes_train[i][0]);
//			cv::Mat img = dlib::toMat(images_train[i]);
//			//cv::rectangle(img, rectTmp, cv::Scalar(0, 255, 0));
//			cv::Mat obj = img(rectTmp).clone();
//			string objname = "E:\\DataSets\\litter\\src_V2_pos_fromtrain\\" + to_string(i) + ".jpg";
//			cv::imwrite(objname, obj);
//
//			//int top = obj_boxes_train[i][0].tl_corner.left();
//			//cout << top << endl;
//			//int left = obj_boxes_train[i][0].left;
//			//int width = obj_boxes_train[i][0].right;
//			//int height = obj_boxes_train[i][0].bottom;
//			//cout << top << " " << left << " " << width << " " << height << endl;
//			//cv::imshow("ss",img);
//			//cv::waitKey();
//
//			std::vector<cv::Rect> rects;
//			for (int y = 0; y < img.cols - 80; y = y + 80)
//			{
//				for (int x = 0; x < img.rows - 80; x = x + 80)
//				{
//					cv::Rect rectTmp2(x, y, 80, 80);
//
//					if (!rectA_intersect_rectB(rectTmp, rectTmp2))//不相交
//					{
//						//rects.push_back(rectTmp2);
//						//cv::rectangle(img, rectTmp2, cv::Scalar(0, 255, 0));
//
//						cv::Mat roi = img(rectTmp2).clone();
//						string name = "E:\\DataSets\\litter\\src_V2_neg_fromtrain\\" + to_string(i) + to_string(x) + "_" + to_string(y) + ".jpg";
//						cv::imwrite(name,roi);
//					}
//				}
//			}
//			//cv::imshow("ss", img);
//			//cv::waitKey();
//		}
//
//		for (int i = 0; i < test_num; i++)
//		{
//			cv::Rect rectTmp = dlibRectangleToOpenCV(obj_boxes_train[i][0]);
//			cv::Mat img = dlib::toMat(images_train[i]);
//			//cv::rectangle(img, rectTmp, cv::Scalar(0, 255, 0));
//			cv::Mat obj = img(rectTmp).clone();
//			string objname = "E:\\DataSets\\litter\\src_V2_pos_fromtest\\" + to_string(i) + ".jpg";
//			cv::imwrite(objname, obj);
//
//			//int top = obj_boxes_train[i][0].tl_corner.left();
//			//cout << top << endl;
//			//int left = obj_boxes_train[i][0].left;
//			//int width = obj_boxes_train[i][0].right;
//			//int height = obj_boxes_train[i][0].bottom;
//			//cout << top << " " << left << " " << width << " " << height << endl;
//			//cv::imshow("ss",img);
//			//cv::waitKey();
//
//			std::vector<cv::Rect> rects;
//			for (int y = 0; y < img.cols - 80; y = y + 80)
//			{
//				for (int x = 0; x < img.rows - 80; x = x + 80)
//				{
//					cv::Rect rectTmp2(x, y, 80, 80);
//
//					if (!rectA_intersect_rectB(rectTmp, rectTmp2))//不相交
//					{
//						//rects.push_back(rectTmp2);
//						//cv::rectangle(img, rectTmp2, cv::Scalar(0, 255, 0));
//
//						cv::Mat roi = img(rectTmp2).clone();
//						string name = "E:\\DataSets\\litter\\src_V2_neg_fromtest\\" + to_string(i) + to_string(x) + "_" + to_string(y) + ".jpg";
//						cv::imwrite(name, roi);
//					}
//				}
//			}
//			//cv::imshow("ss", img);
//			//cv::waitKey();
//		}
//	}
//	else if (flag == 4)//在数据集中裁剪出64x64的正负样本,版本2，不等于就是负样本
//	{
//		int train_num = obj_boxes_train.size();
//		for (int i = 0; i < train_num; i++)
//		{
//			//if (i == 22 || i == 23 || i == 24 || i == 25)
//			//	continue;
//			cv::Rect rectTmp = dlibRectangleToOpenCV(obj_boxes_train[i][0]);
//			cv::Mat img = dlib::toMat(images_train[i]);
//			//cv::rectangle(img, rectTmp, cv::Scalar(0, 255, 0));
//			cv::Mat obj = img(rectTmp).clone();
//			string objname = "E:\\DataSets\\litter\\src_V2_pos_fromtrain_V3\\" + to_string(i) + ".jpg";
//			cv::imwrite(objname, obj);
//			std::vector<cv::Rect> rects;
//			for (int y = 0; y < img.cols - 80+1; y = y + 10)
//			{
//				for (int x = 0; x < img.rows - 80+1; x = x + 10)
//				{
//					cv::Rect rectTmp2(x, y, 80, 80);
//					//cv::rectangle(img, rectTmp2, cv::Scalar(0, 0, 0));
//					if (!rectA_intersect_rectB(rectTmp, rectTmp2))//不相交
//					{
//						cout << "buxiangjiao" << endl;
//						cv::Mat roi = img(rectTmp2).clone();
//						string name = "E:\\DataSets\\litter\\src_V2_neg_fromtrain_V3\\" + to_string(i) + to_string(x) + "_" + to_string(y) + ".jpg";
//						cv::imwrite(name, roi);
//					}
//					else
//					{
//						cv::Rect intersect = rectTmp & rectTmp2;
//						int area = intersect.width*intersect.height;
//						cout << "xiangjiao area is: " << area;
//						if (area < 1500)//稍微相交，也视为负样本
//						{
//							cv::Mat obj = img(rectTmp2).clone();
//							string objname = "E:\\DataSets\\litter\\src_V2_neg_fromtrain_V3\\" + to_string(i) + "_inter" + ".jpg";
//							cv::imwrite(objname, obj);
//							cout << "    -> neg" << endl;
//						}
//						else if (area > 4500)//相交很多，视为正样本
//						{
//							cv::Mat roi = img(rectTmp2).clone();
//							string name = "E:\\DataSets\\litter\\src_V2_pos_fromtrain_V3\\" + to_string(i) + to_string(x) + "_" + to_string(y) + ".jpg";
//							cv::imwrite(name, roi);
//							cout << "    -> pos" << endl;
//						}
//						else
//						{
//							cout << "    -> continue" << endl;
//						}
//					}
//				}
//			}
//			//cv::imshow("ss", img);
//			//cv::waitKey();
//		}
//
//
//		int test_num = obj_boxes_test.size();
//		for (int i = 0; i < test_num; i++)
//		{
//			//if (i == 14)
//			//	continue;
//			cv::Rect rectTmp = dlibRectangleToOpenCV(obj_boxes_test[i][0]);
//			cv::Mat img = dlib::toMat(images_test[i]);
//			//cv::rectangle(img, rectTmp, cv::Scalar(0, 255, 0));
//			cv::Mat obj = img(rectTmp).clone();
//			string objname = "E:\\DataSets\\litter\\src_V2_pos_fromtest_V3\\" + to_string(i) + ".jpg";
//			cv::imwrite(objname, obj);
//			std::vector<cv::Rect> rects;
//			for (int y = 0; y < img.cols - 80+1; y = y + 10)
//			{
//				for (int x = 0; x < img.rows - 80+1; x = x + 10)
//				{
//					cv::Rect rectTmp2(x, y, 80, 80);
//					//cv::rectangle(img, rectTmp2, cv::Scalar(0, 0, 0));
//					if (!rectA_intersect_rectB(rectTmp, rectTmp2))//不相交
//					{
//						cout << "buxiangjiao" << endl;
//						cv::Mat roi = img(rectTmp2).clone();
//						string name = "E:\\DataSets\\litter\\src_V2_neg_fromtest_V3\\" + to_string(i) + to_string(x) + "_" + to_string(y) + ".jpg";
//						cv::imwrite(name, roi);
//						cout << "    -> neg" << endl;
//					}
//					else
//					{
//						cv::Rect intersect = rectTmp & rectTmp2;
//						int area = intersect.width*intersect.height;
//						cout << "xiangjiao area is: " << area;
//
//						if (area < 1500)//稍微相交，也视为负样本
//						{
//							cv::Mat obj = img(rectTmp2).clone();
//							string objname = "E:\\DataSets\\litter\\src_V2_neg_fromtest_V3\\" + to_string(i) + "_inter" + ".jpg";
//							cv::imwrite(objname, obj);
//							cout << "    -> neg" << endl;
//						}
//						else if (area > 4500)//相交很多，视为正样本
//						{
//							cv::Mat roi = img(rectTmp2).clone();
//							string name = "E:\\DataSets\\litter\\src_V2_pos_fromtest_V3\\" + to_string(i) + to_string(x) + "_" + to_string(y) + ".jpg";
//							cv::imwrite(name, roi);
//							cout << "    -> pos" << endl;
//						}
//						else
//						{
//							cout << "    -> continue" << endl;
//						}
//					}
//				}
//			}
//			//cv::imshow("ss", img);
//			//cv::waitKey();
//		}
//
//
//
//
//	}
//	else if (flag == 5)
//	{
//		int train_num = obj_boxes_train.size();
//		for (int i = 0; i < train_num; i++)
//		{
//			//if (i == 22 || i == 23 || i == 24 || i == 25)
//			//	continue;
//			cv::Rect rectTmp = dlibRectangleToOpenCV(obj_boxes_train[i][0]);
//			cv::Mat img = dlib::toMat(images_train[i]);
//			//cv::rectangle(img, rectTmp, cv::Scalar(0, 255, 0));
//			std::vector<cv::Rect> rects;
//			for (int y = 0; y < img.cols - 160 + 1; y = y + 10)
//			{
//				for (int x = 0; x < img.rows - 160 + 1; x = x + 10)
//				{
//					cv::Rect rectTmp2(x, y, 160, 160);
//					//cv::rectangle(img, rectTmp2, cv::Scalar(0, 0, 0));
//					if (!rectA_intersect_rectB(rectTmp, rectTmp2))//不相交
//					{
//						cout << "buxiangjiao" << endl;
//						cv::Mat roi = img(rectTmp2).clone();
//						string name = "E:\\DataSets\\litter\\big_neg1\\" + to_string(i) + to_string(x) + "_" + to_string(y) + ".jpg";
//						cv::imwrite(name, roi);
//					}
//					else
//					{
//						cv::Rect intersect = rectTmp & rectTmp2;
//						int area = intersect.width*intersect.height;
//						cout << "xiangjiao area is: " << area;
//						if (area < 1500)//稍微相交，也视为负样本
//						{
//							cv::Mat obj = img(rectTmp2).clone();
//							string objname = "E:\\DataSets\\litter\\big_neg1\\" + to_string(i) + "_inter" + ".jpg";
//							cv::imwrite(objname, obj);
//							cout << "    -> neg" << endl;
//						}
//					}
//				}
//			}
//		}
//		int test_num = obj_boxes_test.size();
//		for (int i = 0; i < test_num; i++)
//		{
//			cv::Rect rectTmp = dlibRectangleToOpenCV(obj_boxes_test[i][0]);
//			cv::Mat img = dlib::toMat(images_test[i]);
//			//cv::rectangle(img, rectTmp, cv::Scalar(0, 255, 0));
//			std::vector<cv::Rect> rects;
//			for (int y = 0; y < img.cols - 160 + 1; y = y + 10)
//			{
//				for (int x = 0; x < img.rows - 160 + 1; x = x + 10)
//				{
//					cv::Rect rectTmp2(x, y, 160, 160);
//					//cv::rectangle(img, rectTmp2, cv::Scalar(0, 0, 0));
//					if (!rectA_intersect_rectB(rectTmp, rectTmp2))//不相交
//					{
//						cout << "buxiangjiao" << endl;
//						cv::Mat roi = img(rectTmp2).clone();
//						string name = "E:\\DataSets\\litter\\big_neg2\\" + to_string(i) + to_string(x) + "_" + to_string(y) + ".jpg";
//						cv::imwrite(name, roi);
//						cout << "    -> neg" << endl;
//					}
//					else
//					{
//						cv::Rect intersect = rectTmp & rectTmp2;
//						int area = intersect.width*intersect.height;
//						cout << "xiangjiao area is: " << area;
//						if (area < 1500)//稍微相交，也视为负样本
//						{
//							cv::Mat obj = img(rectTmp2).clone();
//							string objname = "E:\\DataSets\\litter\\big_neg2\\" + to_string(i) + "_inter" + ".jpg";
//							cv::imwrite(objname, obj);
//							cout << "    -> neg" << endl;
//						}
//					}
//				}
//			}
//		}
//	}
//
//	system("pause");
//	return 0;
//}
