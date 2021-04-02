#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/opencv.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace dlib;

static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r){
	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

const int flag = 9;

int main(int argc, char** argv)
{
	//训练和检测
	if (flag == 0)
	{
		dlib::array<array2d<unsigned char> > images_train, images_test;
		std::vector<std::vector<rectangle> > obj_boxes_train, obj_boxes_test;
		load_image_dataset(images_train, obj_boxes_train, "E:\\Paper_4_3\\src_4_3_gai\\train.xml");
		load_image_dataset(images_test, obj_boxes_test, "E:\\Paper_4_3\\src_4_3_gai\\test.xml");
		try
		{
			typedef scan_fhog_pyramid<pyramid_down<1> > image_scanner_type;
			image_scanner_type scanner;
			scanner.set_detection_window_size(80, 80);           
			structural_object_detection_trainer<image_scanner_type> trainer(scanner);
			trainer.set_num_threads(1);
			trainer.set_c(4);
			trainer.be_verbose();
			trainer.set_epsilon(0.01);
			object_detector<image_scanner_type> detector = trainer.train(images_train, obj_boxes_train);
			cout << "training results: " << test_object_detection_function(detector, images_train, obj_boxes_train) << endl;
			cout << "testing results:  " << test_object_detection_function(detector, images_test, obj_boxes_test) << endl;
			image_window win;
			for (unsigned long i = 0; i < images_test.size(); ++i)
			{
				std::vector<rectangle> dets = detector(images_test[i]);
				win.clear_overlay();
				win.set_image(images_test[i]);
				win.add_overlay(dets, rgb_pixel(255, 0, 0));
				cout << "Hit enter to process the next image..." << endl;
				cin.get();
			}
			serialize("2019_4_3_80x80_dlibfhogsvm.svm") << detector;
		}
		catch (exception& e)
		{
			cout << "\nexception thrown!" << endl;
			cout << e.what() << endl;
		}
	}
	//用训练好了的分类器检测测试集
	else if (flag == 1)//用训练好了的分类器检测测试集
	{
		dlib::array<array2d<unsigned char> > images_test;
		std::vector<std::vector<rectangle> > obj_boxes_test;
		load_image_dataset(images_test, obj_boxes_test, "E:\\Paper_4_3\\src_4_3_gai\\test.xml");
		typedef scan_fhog_pyramid<pyramid_down<1> > image_scanner_type;
		object_detector<image_scanner_type> detector2;
		deserialize("2019_4_3_80x80_dlibfhogsvm.svm") >> detector2;
		image_window win;
		for (unsigned long i = 0; i < images_test.size(); ++i)
		{
			std::vector<rectangle> dets = detector2(images_test[i]);
			win.clear_overlay();
			win.set_image(images_test[i]);
			win.add_overlay(dets, rgb_pixel(0, 255, 0));
			cout << "Hit enter to process the next image..." << endl;
			cin.get();
		}
	}
	//用训练好了的分类器检测扩充后的数据集，并将能检测出来的图写xml标签
	else if (flag == 2)
	{
		fstream tr;
		tr.open("tr_aug.xml", std::ios::app);
		std::ofstream clc("tr_aug.xml", std::ios::out);
		tr << "<?xml version='1.0' encoding='ISO-8859-1'?>" << std::endl;
		tr << "<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>" << std::endl;
		tr << "<dataset>" << std::endl;
		tr << "<name>imglab dataset</name>" << std::endl;
		tr << "<comment>Created by imglab tool.</comment>" << std::endl;
		tr << "<images>" << std::endl;

		typedef scan_fhog_pyramid<pyramid_down<1> > image_scanner_type;
		object_detector<image_scanner_type> detector2;
		deserialize("2019_4_3_80x80_dlibfhogsvm.svm") >> detector2;
		image_window win;

		for (int n = 1; n <= 1000; n++)
		{
			string name = "E:\\Paper_4_3\\src_4_3_gai\\test2\\" + to_string(n) + ".jpg";
			cv::Mat src = cv::imread(name);
			if (src.empty())
				continue;
			//cv::Mat gray;
			//cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);

			dlib::cv_image<rgb_pixel> dlib_img(src);
			std::vector<rectangle> dets = detector2(dlib_img);
			win.clear_overlay();
			//win.set_image(dlib_img);
			//win.add_overlay(dets, rgb_pixel(0, 255, 0));

			std::vector<cv::Rect> rects;
			for (int j = 0; j < dets.size(); j++)
			{
				cv::Rect rectTmp = dlibRectangleToOpenCV(dets[j]);
				cv::Rect rectTmpNew(rectTmp.x + 1, rectTmp.y + 1, rectTmp.width, rectTmp.height);
				rects.push_back(rectTmpNew);
			}
			//for (int i = 0; i < rects.size(); i++)
			//{
			//	cv::rectangle(src, rects[i], cv::Scalar(0, 0, 255),-1);
			//}

			if (rects.size() == 1 && rects[0].tl().y < 270 && rects[0].tl().x < 270)
			{
				int top = rects[0].tl().y;
				int left = rects[0].tl().x;

				//cout << top << endl;

				tr << "  <image file = 'E:\\Paper_4_3\\src_4_3_gai\\test4\\"<<n<<".jpg'>"<<endl;
				tr << "    <box top='" << top << "' left='" << left << "' width='80' height='80'/>" << endl;
				tr << "  </image>" << endl;

				imwrite("E:\\Paper_4_3\\src_4_3_gai\\test4\\" + to_string(n) + ".jpg", src);
			}

		}
		tr << "</images>" << std::endl;
		tr << "</dataset>" << std::endl;
		tr.close();


	}
	//用训练好了的分类器检测扩充后的数据集，并将能检测出来的图写为VOC-xml标签
	else if (flag == 3)
	{
		typedef scan_fhog_pyramid<pyramid_down<1> > image_scanner_type;
		object_detector<image_scanner_type> detector2;
		deserialize("2019_4_3_80x80_dlibfhogsvm.svm") >> detector2;
		image_window win;

		for (int n = 1; n <= 1663; n++)
		{
			string name = "E:\\2019_4_11\\litter_voc\\" + to_string(n) + ".jpg";
			cv::Mat src = cv::imread(name);
			if (src.empty())
				continue;

			dlib::cv_image<rgb_pixel> dlib_img(src);
			std::vector<rectangle> dets = detector2(dlib_img);
			win.clear_overlay();

			std::vector<cv::Rect> rects;
			for (int j = 0; j < dets.size(); j++)
			{
				cv::Rect rectTmp = dlibRectangleToOpenCV(dets[j]);
				cv::Rect rectTmpNew(rectTmp.x + 1, rectTmp.y + 1, rectTmp.width, rectTmp.height);
				rects.push_back(rectTmpNew);
			}

			if (rects.size() == 1 && rects[0].tl().y < 270 && rects[0].tl().x < 270)
			{
				imwrite("E:\\Paper_4_3\\2019_4_9\\train6\\" + to_string(n) + ".jpg", src);

				int top = rects[0].tl().y;
				int left = rects[0].tl().x;

				fstream tr;
				string picname = std::to_string(n) + ".jpg";
				string xmlname = "E:\\Paper_4_3\\2019_4_9\\train6_xml\\" + std::to_string(n) + ".xml";
				tr.open(xmlname, std::ios::app);
				std::ofstream clctr(xmlname, std::ios::out);

				int xmin = rects[0].tl().x;
				int ymin = rects[0].tl().y;
				int xmax = rects[0].br().x;
				int ymax = rects[0].br().y;

				tr << "<annotation>" << endl;
				tr << "\t<folder>VOC2007</folder>" << endl;
				tr << "\t<filename>"<<picname<<"</filename>" << endl;
				tr << "\t<source>" << endl;
				tr << "\t\t<database>The VOC2007 Database</database>" << endl;
				tr << "\t\t<annotation>PASCAL VOC2007</annotation>" << endl;
				tr << "\t\t<image>flickr</image>" << endl;
				tr << "\t\t<flickrid>194179466</flickrid>" << endl;
				tr << "\t</source>" << endl;
				tr << "\t<owner>" << endl;
				tr << "\t\t<flickrid>monsieurrompu</flickrid>" << endl;
				tr << "\t\t<name>Thom Zemanek</name>" << endl;
				tr << "\t</owner>" << endl;
				tr << "\t<size>" << endl;
				tr << "\t\t<width>350</width>" << endl;
				tr << "\t\t<height>350</height>" << endl;
				tr << "\t\t<depth>3</depth>" << endl;
				tr << "\t</size>" << endl;
				tr << "\t<segmented>0</segmented>" << endl;
				tr << "\t<object>" << endl;
				tr << "\t\t<name>litter</name>" << endl;
				tr << "\t\t<pose>Unspecified</pose>" << endl;
				tr << "\t\t<truncated>1</truncated>" << endl;
				tr << "\t\t<difficult>0</difficult>" << endl;
				tr << "\t\t<bndbox>" << endl;
				tr << "\t\t\t<xmin>" << xmin << "</xmin>" << endl;
				tr << "\t\t\t<ymin>" << ymin << "</ymin>" << endl;
				tr << "\t\t\t<xmax>" << xmax << "</xmax>" << endl;
				tr << "\t\t\t<ymax>" << ymax << "</ymax>" << endl;
				tr << "\t\t</bndbox>" << endl;
				tr << "\t</object>" << endl;
				tr << "</annotation>" << endl;

				tr.close();
			}
		}


		for (int n = 1664; n <= 2186; n++)
		{
			string name = "E:\\Paper_4_3\\2019_4_9\\test5\\" + to_string(n) + ".jpg";
			cv::Mat src = cv::imread(name);
			if (src.empty())
				continue;

			dlib::cv_image<rgb_pixel> dlib_img(src);
			std::vector<rectangle> dets = detector2(dlib_img);
			win.clear_overlay();

			std::vector<cv::Rect> rects;
			for (int j = 0; j < dets.size(); j++)
			{
				cv::Rect rectTmp = dlibRectangleToOpenCV(dets[j]);
				cv::Rect rectTmpNew(rectTmp.x + 1, rectTmp.y + 1, rectTmp.width, rectTmp.height);
				rects.push_back(rectTmpNew);
			}

			if (rects.size() == 1 && rects[0].tl().y < 270 && rects[0].tl().x < 270)
			{
				imwrite("E:\\Paper_4_3\\2019_4_9\\test6\\" + to_string(n) + ".jpg", src);

				int top = rects[0].tl().y;
				int left = rects[0].tl().x;

				fstream tr;
				string picname = std::to_string(n) + ".jpg";
				string xmlname = "E:\\Paper_4_3\\2019_4_9\\test6_xml\\" + std::to_string(n) + ".xml";
				tr.open(xmlname, std::ios::app);
				std::ofstream clctr(xmlname, std::ios::out);

				int xmin = rects[0].tl().x;
				int ymin = rects[0].tl().y;
				int xmax = rects[0].br().x;
				int ymax = rects[0].br().y;

				tr << "<annotation>" << endl;
				tr << "\t<folder>VOC2007</folder>" << endl;
				tr << "\t<filename>" << picname << "</filename>" << endl;
				tr << "\t<source>" << endl;
				tr << "\t\t<database>The VOC2007 Database</database>" << endl;
				tr << "\t\t<annotation>PASCAL VOC2007</annotation>" << endl;
				tr << "\t\t<image>flickr</image>" << endl;
				tr << "\t\t<flickrid>194179466</flickrid>" << endl;
				tr << "\t</source>" << endl;
				tr << "\t<owner>" << endl;
				tr << "\t\t<flickrid>monsieurrompu</flickrid>" << endl;
				tr << "\t\t<name>Thom Zemanek</name>" << endl;
				tr << "\t</owner>" << endl;
				tr << "\t<size>" << endl;
				tr << "\t\t<width>350</width>" << endl;
				tr << "\t\t<height>350</height>" << endl;
				tr << "\t\t<depth>3</depth>" << endl;
				tr << "\t</size>" << endl;
				tr << "\t<segmented>0</segmented>" << endl;
				tr << "\t<object>" << endl;
				tr << "\t\t<name>litter</name>" << endl;
				tr << "\t\t<pose>Unspecified</pose>" << endl;
				tr << "\t\t<truncated>1</truncated>" << endl;
				tr << "\t\t<difficult>0</difficult>" << endl;
				tr << "\t\t<bndbox>" << endl;
				tr << "\t\t\t<xmin>" << xmin << "</xmin>" << endl;
				tr << "\t\t\t<ymin>" << ymin << "</ymin>" << endl;
				tr << "\t\t\t<xmax>" << xmax << "</xmax>" << endl;
				tr << "\t\t\t<ymax>" << ymax << "</ymax>" << endl;
				tr << "\t\t</bndbox>" << endl;
				tr << "\t</object>" << endl;
				tr << "</annotation>" << endl;

				tr.close();
			}
		}


	}
	//直接将已有的XML数据集转为VOC格式
	else if (flag == 9)
	{
		dlib::array<array2d<dlib::bgr_pixel> > images_train, images_test;
		std::vector<std::vector<rectangle> > obj_boxes_train, obj_boxes_test;
		load_image_dataset(images_train, obj_boxes_train, "E:\\Paper_4_3\\src_4_3_gai\\train.xml");
		load_image_dataset(images_test, obj_boxes_test, "E:\\Paper_4_3\\src_4_3_gai\\test.xml");
		for (int i = 0; i < images_train.size(); i++)
		{
			cv::Rect rect = dlibRectangleToOpenCV(obj_boxes_train[i][0]);
			int xmin = rect.tl().x;
			int ymin = rect.tl().y;
			int xmax = rect.br().x;
			int ymax = rect.br().y;

			dlib::array2d<dlib::bgr_pixel> img_bgr;
			cv::Mat img = dlib::toMat(images_train[i]);
			//cv::imshow("ss", img);
			//cv::waitKey();

			string picname = std::to_string(i + 1) + ".jpg";
			cv::imwrite("E:\\2019_4_11\\JPEGImages\\" + picname, img);
			cv::rectangle(img, rect, cv::Scalar(0, 255, 0));
			cv::imwrite("E:\\2019_4_11\\JPEGImages_Groundtruth\\" + picname, img);

			string xmlname = "E:\\2019_4_11\\Annotations\\" + std::to_string(i + 1) + ".xml";
			fstream tr;
			tr.open(xmlname, std::ios::app);
			std::ofstream clctr(xmlname, std::ios::out);
			tr << "<annotation>" << endl;
			tr << "\t<folder>VOC2007</folder>" << endl;
			tr << "\t<filename>" << picname << "</filename>" << endl;
			tr << "\t<source>" << endl;
			tr << "\t\t<database>The VOC2007 Database</database>" << endl;
			tr << "\t\t<annotation>PASCAL VOC2007</annotation>" << endl;
			tr << "\t\t<image>flickr</image>" << endl;
			tr << "\t\t<flickrid>194179466</flickrid>" << endl;
			tr << "\t</source>" << endl;
			tr << "\t<owner>" << endl;
			tr << "\t\t<flickrid>monsieurrompu</flickrid>" << endl;
			tr << "\t\t<name>Thom Zemanek</name>" << endl;
			tr << "\t</owner>" << endl;
			tr << "\t<size>" << endl;
			tr << "\t\t<width>350</width>" << endl;
			tr << "\t\t<height>350</height>" << endl;
			tr << "\t\t<depth>3</depth>" << endl;
			tr << "\t</size>" << endl;
			tr << "\t<segmented>0</segmented>" << endl;
			tr << "\t<object>" << endl;
			tr << "\t\t<name>litter</name>" << endl;
			tr << "\t\t<pose>Unspecified</pose>" << endl;
			tr << "\t\t<truncated>1</truncated>" << endl;
			tr << "\t\t<difficult>0</difficult>" << endl;
			tr << "\t\t<bndbox>" << endl;
			tr << "\t\t\t<xmin>" << xmin << "</xmin>" << endl;
			tr << "\t\t\t<ymin>" << ymin << "</ymin>" << endl;
			tr << "\t\t\t<xmax>" << xmax << "</xmax>" << endl;
			tr << "\t\t\t<ymax>" << ymax << "</ymax>" << endl;
			tr << "\t\t</bndbox>" << endl;
			tr << "\t</object>" << endl;
			tr << "</annotation>" << endl;
			tr.close();
		}
		for (int i = 0; i < images_test.size(); i++)
		{
			cv::Rect rect = dlibRectangleToOpenCV(obj_boxes_test[i][0]);
			int xmin = rect.tl().x;
			int ymin = rect.tl().y;
			int xmax = rect.br().x;
			int ymax = rect.br().y;

			dlib::array2d<dlib::bgr_pixel> img_bgr;
			cv::Mat img = dlib::toMat(images_test[i]);
			//cv::imshow("ss", img);
			//cv::waitKey();

			string picname = std::to_string(i + images_train.size() + 1) + ".jpg";
			cv::imwrite("E:\\2019_4_11\\JPEGImages\\" + picname, img);
			cv::rectangle(img, rect, cv::Scalar(0, 255, 0));
			cv::imwrite("E:\\2019_4_11\\JPEGImages_Groundtruth\\" + picname, img);

			string xmlname = "E:\\2019_4_11\\Annotations\\" + std::to_string(i + images_train.size() + 1) + ".xml";
			fstream tr;
			tr.open(xmlname, std::ios::app);
			std::ofstream clctr(xmlname, std::ios::out);
			tr << "<annotation>" << endl;
			tr << "\t<folder>VOC2007</folder>" << endl;
			tr << "\t<filename>" << picname << "</filename>" << endl;
			tr << "\t<source>" << endl;
			tr << "\t\t<database>The VOC2007 Database</database>" << endl;
			tr << "\t\t<annotation>PASCAL VOC2007</annotation>" << endl;
			tr << "\t\t<image>flickr</image>" << endl;
			tr << "\t\t<flickrid>194179466</flickrid>" << endl;
			tr << "\t</source>" << endl;
			tr << "\t<owner>" << endl;
			tr << "\t\t<flickrid>monsieurrompu</flickrid>" << endl;
			tr << "\t\t<name>Thom Zemanek</name>" << endl;
			tr << "\t</owner>" << endl;
			tr << "\t<size>" << endl;
			tr << "\t\t<width>350</width>" << endl;
			tr << "\t\t<height>350</height>" << endl;
			tr << "\t\t<depth>3</depth>" << endl;
			tr << "\t</size>" << endl;
			tr << "\t<segmented>0</segmented>" << endl;
			tr << "\t<object>" << endl;
			tr << "\t\t<name>litter</name>" << endl;
			tr << "\t\t<pose>Unspecified</pose>" << endl;
			tr << "\t\t<truncated>1</truncated>" << endl;
			tr << "\t\t<difficult>0</difficult>" << endl;
			tr << "\t\t<bndbox>" << endl;
			tr << "\t\t\t<xmin>" << xmin << "</xmin>" << endl;
			tr << "\t\t\t<ymin>" << ymin << "</ymin>" << endl;
			tr << "\t\t\t<xmax>" << xmax << "</xmax>" << endl;
			tr << "\t\t\t<ymax>" << ymax << "</ymax>" << endl;
			tr << "\t\t</bndbox>" << endl;
			tr << "\t</object>" << endl;
			tr << "</annotation>" << endl;
			tr.close();
		}
	}
	system("pause");
	return 0;
}
