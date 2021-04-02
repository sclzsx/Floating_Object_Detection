#include "process_rects.h"

void dlibRectangleToOpenCV(dlib::rectangle r, cv::Rect &output)
{
	output = cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

void get_groundtruth_rects(string inputfile_filename, vector<vector<cv::Rect>> &groundtruths)
{
	dlib::array<dlib::array2d<unsigned char> > images_test;
	std::vector<std::vector<dlib::rectangle> > obj_boxes_test;
	dlib::load_image_dataset(images_test, obj_boxes_test, inputfile_filename);
	for (size_t i = 0; i < images_test.size(); i++)
	{
		vector<cv::Rect> groundtruths_thisframe;
		for (size_t j = 0; j < obj_boxes_test[i].size(); j++)
		{
			cv::Rect tmp;
			dlibRectangleToOpenCV(obj_boxes_test[i][j], tmp);
			groundtruths_thisframe.push_back(tmp);
		}
		groundtruths.push_back(groundtruths_thisframe);
	}
}

void compare_two_rects(cv::Rect rectA, cv::Rect rectB, int &flag)
{
	int area1 = rectA.area();
	int area2 = rectB.area();
	cv::Rect and_rect = rectA & rectB;
	int and_area = and_rect.area();

	if (and_area > 0)
	{
		if (and_area == area1 || and_area == area2)
		{
			flag = 1;//overlap
		}
		else
		{
			flag = 2;//intersect
		}
	}
	else
	{
		flag = 3;//separate
	}
}

void test(vector<cv::Rect> input_rects, int frame_num, vector<cv::Rect> &output_rects)
{
	//vector<std::multimap<cv::Rect, float>> groups;
	//for (int i = 0; i < groundtruths[frame_num].size(); i++)
	//{
	//	std::multimap<cv::Rect, int> true_rects;
	//	//vector<cv::Rect > true_rects;
	//	cv::Rect groundtruth_i = groundtruths[frame_num][i];
	//	for (size_t j = 0; j < input_rects.size(); j++)
	//	{
	//		cv::Rect intersect_rect_tmp = input_rects[j] & groundtruth_i;
	//		float intersect_area = intersect_rect_tmp.area();
	//		float union_area = input_rects[j].area() + groundtruth_i.area() - intersect_area;
	//		float IoU = intersect_area / union_area;
	//		if (IoU > 0)
	//		{
	//			true_rects.insert(std::make_pair(input_rects[j], i));
	//		}
	//		else
	//			true_rects.insert(std::make_pair(input_rects[j], 100));
	//	}
	//}
	//
	//	std::sort(true_rects.begin(), true_rects.end());
	//	cv::Rect nms_rect = *true_rects.end();
	//	output_rects.push_back(nms_rect);
}

void get_nms_rects(vector<cv::Rect> input_rects, int frame_num, vector<cv::Rect> &output_rects)
{
	//for (size_t i = 0; i < groundtruths[frame_num].size(); i++)
	//{
	//	vector<cv::Rect > true_rects;
	//	cv::Rect groundtruth_i = groundtruths[frame_num][i];
	//	for (size_t j = 0; j < input_rects.size(); j++)
	//	{
	//		cv::Rect intersect_rect_tmp = input_rects[j] & groundtruth_i;
	//		float intersect_area = intersect_rect_tmp.area();
	//		float union_area = input_rects[j].area() + groundtruth_i.area() - intersect_area;
	//		float IoU = intersect_area / union_area;

	//		if (IoU > 0)
	//		{
	//			true_rects.push_back(input_rects[j]);
	//		}
	//		else
	//			output_rects.push_back(input_rects[j]);
	//	}

	//	//std::sort(true_rects.begin(), true_rects.end());
	//	//cv::Rect nms_rect = *true_rects.end();
	//	//output_rects.push_back(nms_rect);
	//}
}

void combine_two_methods_rects(vector<cv::Rect> time_rects, vector<cv::Rect> frequency_rects, vector<cv::Rect> &output_rects, float th1, float th2)
{
	output_rects.clear();
	for (size_t j = 0; j < time_rects.size(); j++)
	{
		for (size_t k = 0; k < frequency_rects.size(); k++)
		{
			int position;
			compare_two_rects(time_rects[j], frequency_rects[k], position);
			if (position == 1 || position == 2)
			{
				cv::Rect intersect_rect_tmp = time_rects[j] & frequency_rects[k];
				float intersect_area = intersect_rect_tmp.area();
				float union_area = time_rects[j].area() + frequency_rects[k].area() - intersect_area;
				float IoU = intersect_area / union_area;
				cout << IoU << endl;
				if (IoU < th1)
				{
					continue;
				}
				else
				{
					if (IoU > th2)
					{
						float time_rect_area = time_rects[j].area();
						float frequency_rect_area = frequency_rects[k].area();
						float RP_t = intersect_area / time_rect_area;
						float RP_f = intersect_area / frequency_rect_area;
						if (RP_t >= RP_f)
						{
							output_rects.push_back(time_rects[j]);
						}
						else
						{
							output_rects.push_back(frequency_rects[k]);
						}
					}
					else
					{
						output_rects.push_back(intersect_rect_tmp);
					}
				}
			}
		}
	}
}

void get_confidences_of_rects(vector<cv::Rect> input_rects, int frame_num, vector<float> &confidences)
{
	//confidences.clear();
	//for (size_t j = 0; j < input_rects.size(); j++)
	//{
	//	vector<float> ious_tmp;
	//	ious_tmp.clear();
	//	ious_tmp.push_back(0);
	//	for (size_t k = 0; k < groundtruths[frame_num].size(); k++)
	//	{
	//		cv::Rect groundtruth_k = groundtruths[frame_num][k];
	//		cv::Rect intersect_rect_tmp = input_rects[j] & groundtruth_k;
	//		float intersect_area = intersect_rect_tmp.area();
	//		float union_area = input_rects[j].area() + groundtruth_k.area() - intersect_area;
	//		float IoU = intersect_area / union_area;
	//		ious_tmp.push_back(IoU);
	//	}
	//	std::sort(ious_tmp.begin(), ious_tmp.end());
	//	float max = *ious_tmp.end();
	//	confidences.push_back(max);
	//}
}

void get_correct_rects(vector<cv::Rect> input_rects, vector<cv::Rect> thisframe_groundtruths, int thresh, vector<cv::Rect> &output_rects)
{
	for (size_t j = 0; j < input_rects.size(); j++)
	{
		for (size_t k = 0; k < thisframe_groundtruths.size(); k++)
		{
			cv::Rect groundtruth_k = thisframe_groundtruths[k];
			int position;
			compare_two_rects(input_rects[j], groundtruth_k, position);
			if (position == 1 || position == 2)
			{
				cv::Rect intersect_rect_tmp = input_rects[j] & groundtruth_k;
				float intersect_area = intersect_rect_tmp.area();
				float union_area = input_rects[j].area() + groundtruth_k.area() - intersect_area;
				float IoU = intersect_area / union_area;
				if (IoU >= thresh)
				{
					output_rects.push_back(input_rects[j]);
				}
			}
		}
	}
}

void get_mid_rect(vector<cv::Rect> input_rects, cv::Rect &output_rect)
{
	int x = 0, y = 0;
	for (int i = 0; i < input_rects.size(); i++)
	{
		x = x + input_rects[i].tl().x + winsize / 2;
		y = y + input_rects[i].tl().y + winsize / 2;
	}
	x = x / input_rects.size();
	y = y / input_rects.size();
	output_rect = cv::Rect(x - winsize / 2, y - winsize / 2, winsize, winsize);
}

void dbscan_process(vector<cv::Rect> input_rects, vector<cv::Rect> &output_rects)
{
	vector<dbscanPoint> dataset;
	for (int i = 0; i < input_rects.size(); i++)
	{
		dbscanPoint p;
		p.x = input_rects[i].tl().x + winsize / 2;
		p.y = input_rects[i].tl().y + winsize / 2;
		p.lable = -1;
		dataset.push_back(p);
	}
	int c = dbscan(dataset, 40, 1);
	for (int i = 0; i < dataset.size(); i++)
	{
		cout << "(" << dataset[i].x << "," << dataset[i].y << ") " << dataset[i].lable << endl;
	}

	int group_num = 1;
	int prelable = 1;
	for (int i = 0; i < dataset.size(); i++)
	{
		if (dataset[i].lable != prelable)
		{
			group_num++;
			prelable++;
		}
	}

	for (int i = 1; i <= group_num; i++)
	{
		cv::Rect mid_rect;

		vector<cv::Rect> tmp;
		for (int j = 0; j < dataset.size(); j++)
		{
			if (dataset[j].lable == group_num)
			{
				tmp.push_back(input_rects[i]);
			}
		}

		get_mid_rect(tmp, mid_rect);
		output_rects.push_back(mid_rect);
	}
}

void show_rects(string winname, Mat input, vector<cv::Rect> input_rects)
{
	Mat tmp = input.clone();
	for (int i = 0; i < input_rects.size(); i++)
	{
		cv::Rect rectTmp = input_rects[i];
		cv::rectangle(tmp, rectTmp, cv::Scalar(255, 0, 0), 1);
	}
	imshow(winname, tmp);
	cv::imwrite("tmp.jpg", tmp);
}

void mergeRects(vector<cv::Rect> rects, cv::Rect &output)
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
	int width = brx - tlx;
	int height = bry - tly;
	cv::Point center(tlx + width / 2, tly + height / 2);
	output = cv::Rect(center.x - winsize / 2, center.y - winsize / 2, winsize, winsize);
}


float getDistance(cv::Point A, cv::Point B)
{
	float dis = (A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y);
	//dis =sqrt(dis);
	return dis;
}
