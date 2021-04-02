#include "lrr.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <numeric>
//#include <string>
using std::vector;
using cv::Mat;
using cv::Rect;
using std::cout;
using std::endl;

void load_images(const std::string dirname, std::vector< cv::Mat > &img_lst)
{
	std::vector<  std::string > files;
	cv::glob(dirname, files);
	for (size_t i = 0; i < files.size(); ++i)
	{
		Mat img = cv::imread(files[i]); // load the image
		if (img.empty())            // invalid image, skip it.
		{
			std::cout << files[i] << " is invalid!" << std::endl;
			continue;
		}
		img_lst.push_back(img);
	}
}

void MyGammaCorrection(Mat src, Mat& dst, float fGamma)
{
	CV_Assert(src.data);
	CV_Assert(src.depth() != sizeof(uchar));

	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}

	dst = src.clone();
	const int channels = dst.channels();
	switch (channels)
	{
	case 1:
	{

		cv::MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			*it = lut[(*it)];

		break;
	}
	case 3:
	{

		cv::MatIterator_<cv::Vec3b> it, end;
		for (it = dst.begin<cv::Vec3b>(), end = dst.end<cv::Vec3b>(); it != end; it++)
		{
			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}

		break;

	}
	}
}

void low_rank_decomposition(Mat input, Mat &z, Mat &e)
{
	MatrixXd mat(input.rows, input.cols);
	cv::cv2eigen(input, mat);

	double lambda = 0.01;
	LowRankRepresentation mylrr;
	std::vector<MatrixXd> ZE = mylrr.result(mat, lambda);

	cv::eigen2cv(ZE[0], z);
	cv::eigen2cv(ZE[1], e);
}

void fastdft_and_highpass(cv::Mat src_img, cv::Mat &real_img, cv::Mat &ima_img)
{
	if (src_img.channels() != 1)
		cvtColor(src_img, src_img, cv::COLOR_RGB2GRAY);

	src_img.convertTo(src_img, CV_64FC1);

	int oph = cv::getOptimalDFTSize(src_img.rows);
	int opw = cv::getOptimalDFTSize(src_img.cols);
	Mat padded;
	copyMakeBorder(src_img, padded, 0, oph - src_img.rows, 0, opw - src_img.cols,
		cv::BORDER_CONSTANT, cv::Scalar::all(0));
	padded.convertTo(padded, CV_64FC1);

	for (int i = 0; i < padded.rows; i++)//中心化操作
	{
		double *ptr = padded.ptr<double>(i);
		for (int j = 0; j < padded.cols; j++)	ptr[j] *= pow(-1, i + j);
	}

	Mat plane[] = { padded,Mat::zeros(padded.size(),CV_64FC1) };
	Mat complexImg;
	merge(plane, 2, complexImg);
	dft(complexImg, complexImg);


	Mat gaussianSharpen(padded.size(), CV_64FC2);
	double D0 = 2 * 10 * 10;
	for (int i = 0; i < padded.rows; i++)
	{
		double*q = gaussianSharpen.ptr<double>(i);//高通，高斯锐化
		for (int j = 0; j < padded.cols; j++)
		{
			double d = pow(i - padded.rows / 2, 2) + pow(j - padded.cols / 2, 2);
			q[2 * j] = 1 - expf(-d / D0);
			q[2 * j + 1] = 1 - expf(-d / D0);
		}
	}
	multiply(complexImg, gaussianSharpen, gaussianSharpen);
	split(gaussianSharpen, plane);

	//split(complexImg, plane);
	plane[0].copyTo(real_img);
	plane[1].copyTo(ima_img);
}

void one_amplitude(Mat complex_r, Mat &complex_i, Mat &dst)
{
	Mat temp[] = { Mat::zeros(complex_r.size(),CV_64FC1), Mat::zeros(complex_r.size(),CV_64FC1) };
	float realv = 0.0, imaginv = 0.0;
	for (int i = 0; i < complex_r.cols; i++) {
		for (int j = 0; j < complex_r.rows; j++) {
			realv = complex_r.at<float>(i, j);
			imaginv = complex_i.at<float>(i, j);
			float distance = sqrt(realv*realv + imaginv * imaginv);
			temp[0].at<float>(i, j) = realv / distance;
			temp[1].at<float>(i, j) = imaginv / distance;
		}
	}
	merge(temp, 2, dst);
}

cv::Mat fourior_inverser(Mat &_complexim)
{
	Mat dst;
	Mat iDft[] = { Mat::zeros(_complexim.size(),CV_64FC1),Mat::zeros(_complexim.size(),CV_64FC1) };//创建两个通道，类型为float，大小为填充后的尺寸
	idft(_complexim, _complexim);//傅立叶逆变换
	split(_complexim, iDft);//结果貌似也是复数
	magnitude(iDft[0], iDft[1], dst);//分离通道，主要获取0通道
//    dst += Scalar::all(1);                    // switch to logarithmic scale
//    log(dst, dst);
	//归一化处理，float类型的显示范围为0-255,255为白色，0为黑色
	normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
	return dst;
}

void get_sobel_map(Mat input, Mat &output)
{
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Sobel(input, grad_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	Sobel(input, grad_y, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, output);
}

void get_saliency_map(Mat input, Mat &out_chushi, Mat &output)
{
	Mat in = input.clone();
	cv::imshow("in", in);

	Mat ga;
	MyGammaCorrection(in, ga, 6);
	//imshow("gamma", ga);

	Mat real, imag;
	fastdft_and_highpass(ga, real, imag);
	Mat img_range;
	one_amplitude(real, imag, img_range);
	Mat tmp = fourior_inverser(img_range);
	Mat chushi = tmp(cv::Rect(0, 0, input.cols, input.rows));
	//imshow("chushi", chushi);

	cv::normalize(chushi, out_chushi, 0, 255, cv::NORM_MINMAX);
	out_chushi.convertTo(out_chushi, CV_8UC1);
	threshold(out_chushi, out_chushi, 10, 255, cv::THRESH_BINARY);

	int desize = 50;
	Mat chushi_clone1 = chushi.clone();
	cv::resize(chushi_clone1, chushi_clone1, cv::Size(desize, desize), cv::INTER_AREA);
	Mat low_rank_part, sparse_part;
	low_rank_decomposition(chushi_clone1, low_rank_part, sparse_part);
	Mat quanju = sparse_part.clone();
	cv::resize(quanju, quanju, input.size(), cv::INTER_CUBIC);
	//imshow("quanju", quanju);

	Mat chushi_clone2 = chushi.clone();
	int blocknum = 0;
	int blocksize = 50;
	int num = (chushi_clone2.cols / blocksize)*(chushi_clone2.cols / blocksize);
	Mat Y = Mat::zeros(blocksize, num, CV_64FC1);
	vector<Mat> pk;
	for (int y = 0; y < chushi_clone2.cols - blocksize + 1; y = y + blocksize)
	{
		for (int x = 0; x < chushi_clone2.rows - blocksize + 1; x = x + blocksize)
		{
			cv::Rect rectTmp(y, x, blocksize, blocksize);
			Mat p = chushi_clone2(rectTmp).clone();
			pk.push_back(p);
			vector<double> col;
			for (int i = 0; i < p.rows; i++)
			{
				vector<int> row;
				for (int j = 0; j < p.cols; j++)
				{
					int tmp = p.at<int>(i, j);
					row.push_back(tmp);
				}
				int sum = std::accumulate(std::begin(row), std::end(row), 0);
				double mean = (double)sum / row.size();
				col.push_back(mean);
			}

			for (int j = 0; j < col.size(); j++)
			{
				Y.at<double>(j, blocknum) = col[j];
			}

			blocknum++;
		}
	}
	Mat low_rank_part1, sparse_part1;
	low_rank_decomposition(Y, low_rank_part1, sparse_part1);
	Mat jubu = sparse_part1.clone();
	vector<double> S;
	for (int k = 0; k < jubu.cols; k++)
	{
		Mat col_tmp = Mat::zeros(blocksize, 1, CV_64FC1);
		for (int t = 0; t < col_tmp.rows; t++)
		{
			col_tmp.at<double>(t, 0) = (double)jubu.at<double>(t, k);
		}
		double sp = cv::norm(col_tmp);
		S.push_back(sp);
		//cout << sp << endl;
	}
	float d1 = 0.8, d2 = 0.2;
	float u1 = 0.9, u2 = 0.1;
	int block = 0;
	for (int y = 0; y < chushi_clone2.cols - blocksize + 1; y = y + blocksize)
	{
		for (int x = 0; x < chushi_clone2.rows - blocksize + 1; x = x + blocksize)
		{
			cv::Rect rectTmp(y, x, blocksize, blocksize);
			//Mat p = chushi_clone2(rectTmp);
			if (S[block] > d2 && S[block] <= d1)
			{
				chushi_clone2(rectTmp) = u1 * pk[block];
			}
			else if (S[block] <= d2)
			{
				chushi_clone2(rectTmp) = u2 * pk[block];
			}
			else
			{
				continue;
			}
			block++;
		}
	}
	jubu = chushi_clone2;
	//imshow("jubu", jubu);

	Mat final = quanju & jubu;
	cv::normalize(final, final, 0, 255, cv::NORM_MINMAX);
	final.convertTo(final, CV_8UC1);
	//imshow("final", final);

	cv::medianBlur(final, final, 5);//必须得有
	//imshow("final_filtered", final);

	//imshow("final_morphed", final);
	////cv::imwrite("final.jpg", final);
	threshold(final, final, 0, 255, cv::THRESH_OTSU);
	//imshow("thresh", final);
	output = final.clone();
}

void get_saliency_rects(Mat input, vector<cv::Rect> &output_rects)
{
	Mat saliency_in = input.clone();

	if (input.channels() != 1)
		cvtColor(input, input, cv::COLOR_RGB2GRAY);
	if (input.type() != CV_8UC1)
		input.convertTo(input, CV_8UC1);

	Mat bin_sr;
	threshold(input, bin_sr, 0, 255, cv::THRESH_OTSU);
	Mat element_sr = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
	morphologyEx(bin_sr, bin_sr, cv::MORPH_CLOSE, element_sr);
	morphologyEx(bin_sr, bin_sr, cv::MORPH_DILATE, element_sr);
	vector<vector<cv::Point> >contours_sr;
	vector<cv::Vec4i> hierarchy_sr;
	findContours(bin_sr, contours_sr, hierarchy_sr, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours_sr.size(); i++)
	{
		int areaTmp = contourArea(contours_sr[i]);
		if (areaTmp > 500 && areaTmp < 10000)
		{
			cv::Rect rectTmp = boundingRect(contours_sr[i]);
			output_rects.push_back(rectTmp);
		}
	}
	//Mat tmp = Mat::zeros (bin_sr.size(),CV_8UC3);
	//cv::drawContours(tmp,contours_sr,-1,cv::Scalar(0,255,0));
	//cv::imshow("all_contours",tmp);
	//cv::imwrite("con.jpg",tmp);
}

void show_rects(std::string winname, Mat input, vector<cv::Rect> input_rects)
{
	Mat tmp = input.clone();
	for (int i = 0; i < input_rects.size(); i++)
	{
		cv::Rect rectTmp = input_rects[i];
		cv::rectangle(tmp, rectTmp, cv::Scalar(255, 0, 0), 1);
	}
	imshow(winname, tmp);
	//cv::imwrite("E:\\Paper_4_3\\re_" + std::to_string(i) + ".jpg", tmp);
}

int main(int argc, char *argv[])
{

	for (int i = 1;i<=100;i++)
	{
		Mat in = cv::imread("C:\\Users\\sx\\Desktop\\img2\\" + std::to_string(i) + ".jpg");
		if (in.empty())
			continue;

		//Mat in = cv::imread("94.jpg");

		Mat saliency_map;
		Mat out_chushi;
		get_saliency_map(in, out_chushi, saliency_map);
		//cv::imshow("ss", saliency_map);
		cv::imwrite("C:\\Users\\sx\\Desktop\\img2\\chushi_" + std::to_string(i) + ".jpg", out_chushi);
		cv::imwrite("C:\\Users\\sx\\Desktop\\img2\\sa_" + std::to_string(i) + ".jpg", saliency_map);

		vector<cv::Rect> myphase_rects;
		get_saliency_rects(saliency_map, myphase_rects);
		//show_rects("sx", in, myphase_rects);
		for (int i = 0; i < myphase_rects.size(); i++)
		{
			cv::Rect rectTmp = myphase_rects[i];
			cv::rectangle(in, rectTmp, cv::Scalar(255, 0, 0), 1);
		}
		//imshow("rect", in);
		cv::imwrite("C:\\Users\\sx\\Desktop\\img2\\re_" + std::to_string(i) + ".jpg", in);

		cv::waitKey(0);
	}
	cout << "fin" << endl;
	
	return 0;
}




























//////////////
//////////////
//////////////
//////////////
//////////////
//////////////
//////////////
//////////////
//////////////#include "lrr.h"
//////////////#include <opencv2/opencv.hpp>
//////////////#include <opencv2/core/eigen.hpp>
//////////////#include <iostream>
//////////////#include <numeric>
//////////////using std::vector;
//////////////using cv::Mat;
//////////////using cv::Rect;
//////////////
//////////////void load_images(const std::string dirname, std::vector< cv::Mat > &img_lst)
//////////////{
//////////////	std::vector<  std::string > files;
//////////////	cv::glob(dirname, files);
//////////////	for (size_t i = 0; i < files.size(); ++i)
//////////////	{
//////////////		Mat img = cv::imread(files[i]); // load the image
//////////////		if (img.empty())            // invalid image, skip it.
//////////////		{
//////////////			std::cout << files[i] << " is invalid!" << std::endl;
//////////////			continue;
//////////////		}
//////////////		img_lst.push_back(img);
//////////////	}
//////////////}
//////////////
//////////////void MyGammaCorrection(Mat src, Mat& dst, float fGamma)
//////////////{
//////////////	CV_Assert(src.data);
//////////////	CV_Assert(src.depth() != sizeof(uchar));
//////////////
//////////////	unsigned char lut[256];
//////////////	for (int i = 0; i < 256; i++)
//////////////	{
//////////////		lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
//////////////	}
//////////////
//////////////	dst = src.clone();
//////////////	const int channels = dst.channels();
//////////////	switch (channels)
//////////////	{
//////////////	case 1:
//////////////	{
//////////////
//////////////		cv::MatIterator_<uchar> it, end;
//////////////		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
//////////////			*it = lut[(*it)];
//////////////
//////////////		break;
//////////////	}
//////////////	case 3:
//////////////	{
//////////////
//////////////		cv::MatIterator_<cv::Vec3b> it, end;
//////////////		for (it = dst.begin<cv::Vec3b>(), end = dst.end<cv::Vec3b>(); it != end; it++)
//////////////		{
//////////////			(*it)[0] = lut[((*it)[0])];
//////////////			(*it)[1] = lut[((*it)[1])];
//////////////			(*it)[2] = lut[((*it)[2])];
//////////////		}
//////////////
//////////////		break;
//////////////
//////////////	}
//////////////	}
//////////////}
//////////////
//////////////void get_saliency_rects(Mat input, std::vector<cv::Rect> &output_rects)
//////////////{
//////////////	Mat saliency_in = input.clone();
//////////////	Mat saliency;
//////////////
//////////////	get_myphase_saliency2(saliency_in, saliency);
//////////////
//////////////	if (saliency.channels() != 1)
//////////////		cvtColor(saliency, saliency, cv::COLOR_RGB2GRAY);
//////////////	if (saliency.type() != CV_8UC1)
//////////////		saliency.convertTo(saliency, CV_8UC1);
//////////////
//////////////	Mat bin_sr;
//////////////	threshold(saliency, bin_sr, 0, 255, cv::THRESH_OTSU);
//////////////	Mat element_sr = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
//////////////	morphologyEx(bin_sr, bin_sr, cv::MORPH_CLOSE, element_sr);
//////////////	morphologyEx(bin_sr, bin_sr, cv::MORPH_DILATE, element_sr);
//////////////	std::vector<std::vector<cv::Point> >contours_sr;
//////////////	std::vector<cv::Vec4i> hierarchy_sr;
//////////////	findContours(bin_sr, contours_sr, hierarchy_sr, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
//////////////	for (int i = 0; i < contours_sr.size(); i++)
//////////////	{
//////////////		int areaTmp = contourArea(contours_sr[i]);
//////////////		if (areaTmp > 10 && areaTmp < 10000)
//////////////		{
//////////////			cv::Rect rectTmp = boundingRect(contours_sr[i]);
//////////////			output_rects.push_back(rectTmp);
//////////////		}
//////////////	}
//////////////}
//////////////
//////////////void low_rank_decomposition(Mat input, Mat &z, Mat &e)
//////////////{
//////////////	MatrixXd mat(input.rows, input.cols);
//////////////	cv::cv2eigen(input, mat);
//////////////
//////////////	double lambda = 0.01;
//////////////	LowRankRepresentation mylrr;
//////////////	std::vector<MatrixXd> ZE = mylrr.result(mat, lambda);
//////////////
//////////////	cv::eigen2cv(ZE[0], z);
//////////////	cv::eigen2cv(ZE[1], e);
//////////////}
//////////////
//////////////void fast_dft(cv::Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img)
//////////////{
//////////////	if (src_img.channels() != 1)
//////////////		cvtColor(src_img, src_img, cv::COLOR_RGB2GRAY);
//////////////
//////////////	src_img.convertTo(src_img, CV_64FC1);
//////////////
//////////////	///////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
//////////////	int oph = cv::getOptimalDFTSize(src_img.rows);
//////////////	int opw = cv::getOptimalDFTSize(src_img.cols);
//////////////	Mat padded;
//////////////	copyMakeBorder(src_img, padded, 0, oph - src_img.rows, 0, opw - src_img.cols,
//////////////		cv::BORDER_CONSTANT, cv::Scalar::all(0));
//////////////
//////////////	Mat temp[] = { padded, Mat::zeros(padded.size(),CV_64FC1) };
//////////////	Mat complexI;
//////////////	merge(temp, 2, complexI);
//////////////	dft(complexI, complexI);    //傅里叶变换
//////////////	split(complexI, temp);      //显示频谱图
//////////////	temp[0].copyTo(real_img);
//////////////	temp[1].copyTo(ima_img);
//////////////}
//////////////
//////////////void one_amplitude(Mat &complex_r, Mat &complex_i, Mat &dst)
//////////////{
//////////////	Mat temp[] = { Mat::zeros(complex_r.size(),CV_64FC1), Mat::zeros(complex_r.size(),CV_64FC1) };
//////////////	float realv = 0.0, imaginv = 0.0;
//////////////	for (int i = 0; i < complex_r.cols; i++) {
//////////////		for (int j = 0; j < complex_r.rows; j++) {
//////////////			realv = complex_r.at<float>(i, j);
//////////////			imaginv = complex_i.at<float>(i, j);
//////////////			float distance = sqrt(realv*realv + imaginv * imaginv);
//////////////			temp[0].at<float>(i, j) = realv / distance;
//////////////			temp[1].at<float>(i, j) = imaginv / distance;
//////////////		}
//////////////	}
//////////////	merge(temp, 2, dst);
//////////////}
//////////////
//////////////cv::Mat fourior_inverser(Mat &_complexim)
//////////////{
//////////////	Mat dst;
//////////////	Mat iDft[] = { Mat::zeros(_complexim.size(),CV_64FC1),Mat::zeros(_complexim.size(),CV_64FC1) };//创建两个通道，类型为float，大小为填充后的尺寸
//////////////	idft(_complexim, _complexim);//傅立叶逆变换
//////////////	split(_complexim, iDft);//结果貌似也是复数
//////////////	magnitude(iDft[0], iDft[1], dst);//分离通道，主要获取0通道
////////////////    dst += Scalar::all(1);                    // switch to logarithmic scale
////////////////    log(dst, dst);
//////////////	//归一化处理，float类型的显示范围为0-255,255为白色，0为黑色
//////////////	normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
//////////////	return dst;
//////////////}
//////////////
//////////////void get_myphase_saliency2(Mat input, Mat &output)
//////////////{
//////////////	Mat myphase_in = input.clone();
//////////////	MyGammaCorrection(myphase_in, myphase_in, 5);
//////////////	Mat grad_x, grad_y;
//////////////	Mat abs_grad_x, abs_grad_y;
//////////////	Sobel(myphase_in, grad_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
//////////////	convertScaleAbs(grad_x, abs_grad_x);
//////////////	Sobel(myphase_in, grad_y, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
//////////////	convertScaleAbs(grad_y, abs_grad_y);
//////////////	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, myphase_in);
//////////////	imshow("zengqiang", myphase_in);
//////////////	cv::imwrite("zengqiang.jpg", myphase_in);
//////////////
//////////////	Mat real, imag;
//////////////	fast_dft(myphase_in, real, imag);
//////////////	Mat img_range;
//////////////	one_amplitude(real, imag, img_range);
//////////////	Mat tmp = fourior_inverser(img_range);
//////////////	Mat chushi = tmp(cv::Rect(0, 0, input.cols, input.rows));
//////////////	//imshow("chushi", chushi);
//////////////
//////////////
//////////////	int desize = 40;
//////////////	Mat chushi_clone1 = chushi.clone();
//////////////	cv::resize(chushi_clone1, chushi_clone1, cv::Size(desize, desize), cv::INTER_AREA);
//////////////	Mat low_rank_part, sparse_part;
//////////////	low_rank_decomposition(chushi_clone1, low_rank_part, sparse_part);
//////////////	Mat quanju = chushi_clone1 - low_rank_part;
//////////////	//Mat quanju = sparse_part.clone();
//////////////	cv::resize(quanju, quanju, input.size(), cv::INTER_CUBIC);
//////////////	//imshow("quanju", quanju);
//////////////
//////////////
//////////////	Mat chushi_clone2 = chushi.clone();
//////////////	int blocknum = 0;
//////////////	int blocksize = 50;
//////////////	int num = (chushi_clone2.cols / blocksize)*(chushi_clone2.cols / blocksize);
//////////////	Mat Y = Mat::zeros(blocksize, num, CV_64FC1);
//////////////	vector<Mat> pk;
//////////////	for (int y = 0; y < chushi_clone2.cols - blocksize + 1; y = y + blocksize)
//////////////	{
//////////////		for (int x = 0; x < chushi_clone2.rows - blocksize + 1; x = x + blocksize)
//////////////		{
//////////////			cv::Rect rectTmp(y, x, blocksize, blocksize);
//////////////			Mat p = chushi_clone2(rectTmp).clone();
//////////////			pk.push_back(p);
//////////////			vector<double> col;
//////////////			for (int i = 0; i < p.rows; i++)
//////////////			{
//////////////				vector<int> row;
//////////////				for (int j = 0; j < p.cols; j++)
//////////////				{
//////////////					int tmp = p.at<int>(i, j);
//////////////					row.push_back(tmp);
//////////////				}
//////////////				int sum = std::accumulate(std::begin(row), std::end(row), 0);
//////////////				double mean = (double)sum / row.size();
//////////////				col.push_back(mean);
//////////////			}
//////////////
//////////////			for (int j = 0; j < col.size(); j++)
//////////////			{
//////////////				Y.at<double>(j, blocknum) = col[j];
//////////////			}
//////////////
//////////////			blocknum++;
//////////////		}
//////////////	}
//////////////	Mat low_rank_part1, sparse_part1;
//////////////	low_rank_decomposition(Y, low_rank_part1, sparse_part1);
//////////////	Mat jubu = Y - low_rank_part1;
//////////////	//Mat jubu = sparse_part1.clone();
//////////////	vector<double> S;
//////////////	for (int k = 0; k < jubu.cols; k++)
//////////////	{
//////////////		Mat col_tmp = Mat::zeros(blocksize, 1, CV_64FC1);
//////////////		for (int t = 0; t < col_tmp.rows; t++)
//////////////		{
//////////////			col_tmp.at<double>(t, 0) = (double)jubu.at<double>(t, k);
//////////////		}
//////////////		double sp = cv::norm(col_tmp);
//////////////		S.push_back(sp);
//////////////		//cout << sp << endl;
//////////////	}
//////////////	float d1 = 0.8, d2 = 0.2;
//////////////	float u1 = 0.8, u2 = 0.2;
//////////////	int block = 0;
//////////////	for (int y = 0; y < chushi_clone2.cols - blocksize + 1; y = y + blocksize)
//////////////	{
//////////////		for (int x = 0; x < chushi_clone2.rows - blocksize + 1; x = x + blocksize)
//////////////		{
//////////////			cv::Rect rectTmp(y, x, blocksize, blocksize);
//////////////			//Mat p = chushi_clone2(rectTmp);
//////////////			if (S[block] > d2 && S[block] <= d1)
//////////////			{
//////////////				chushi_clone2(rectTmp) = u1 * pk[block];
//////////////			}
//////////////			else if (S[block] <= d2)
//////////////			{
//////////////				chushi_clone2(rectTmp) = u2 * pk[block];
//////////////			}
//////////////			else
//////////////			{
//////////////				continue;
//////////////			}
//////////////			block++;
//////////////		}
//////////////	}
//////////////	jubu = chushi_clone2;
//////////////	//imshow("jubu", jubu);
//////////////
//////////////
//////////////	float r1 = 0.6, r2 = 0.4;
//////////////	Mat final = quanju * 0.6 + jubu * 0.4;
//////////////
//////////////
//////////////	//Mat e1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
//////////////	//Mat e2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
//////////////
//////////////	//cv::normalize(chushi, chushi, 0, 255, cv::NORM_MINMAX);
//////////////	//chushi.convertTo(chushi, CV_8UC1);
//////////////	//threshold(chushi, final, 80, 255, cv::THRESH_BINARY);
//////////////	//imshow("chushi", chushi);
//////////////	//cv::imwrite("chushi.jpg", chushi);
//////////////
//////////////	//cv::normalize(jubu, jubu, 0, 255, cv::NORM_MINMAX);
//////////////	//jubu.convertTo(jubu, CV_8UC1);
//////////////	//threshold(jubu, jubu, 100, 255, cv::THRESH_BINARY);
//////////////	////morphologyEx(jubu, jubu, cv::MORPH_ERODE, e1);
//////////////	//morphologyEx(jubu, jubu, cv::MORPH_DILATE, e2);
//////////////	//imshow("jubu", jubu);
//////////////	//cv::imwrite("jubu.jpg", jubu);
//////////////	//
//////////////
//////////////
//////////////	//cv::normalize(quanju, quanju, 0, 255, cv::NORM_MINMAX);
//////////////	//quanju.convertTo(quanju, CV_8UC1);
//////////////	//threshold(quanju, quanju, 100, 255, cv::THRESH_BINARY);
//////////////	//morphologyEx(quanju, quanju, cv::MORPH_ERODE, e1);
//////////////	//morphologyEx(quanju, quanju, cv::MORPH_DILATE, e2);
//////////////	//imshow("quanju", quanju);
//////////////	//cv::imwrite("quanju.jpg", quanju);
//////////////
//////////////	//cv::normalize(final, final, 0, 255, cv::NORM_MINMAX);
//////////////	//final.convertTo(final, CV_8UC1);
//////////////	//threshold(final, final, 120, 255, cv::THRESH_BINARY);
//////////////	//morphologyEx(final, final, cv::MORPH_ERODE, e1);
//////////////	//morphologyEx(final, final, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)));
//////////////	//imshow("final", final);
//////////////	//cv::imwrite("final.jpg", final);
//////////////
//////////////
//////////////
//////////////	output = final.clone();
//////////////}
//////////////
////////////////void get_phase_map(Mat img, Mat &output)
////////////////{
////////////////	//cv::Mat img = cv::imread("test.jpg");
////////////////	if (img.channels() != 1)
////////////////		cvtColor(img, img, cv::COLOR_RGB2GRAY);
////////////////	float ratio = 64.0 / img.cols;
////////////////	cv::resize(img, img, cv::Size(img.cols*ratio, img.rows*ratio));
////////////////
////////////////	cv::Mat planes[] = { cv::Mat_<float>(img), cv::Mat::zeros(img.size(), CV_32F) };
////////////////	cv::Mat complexImg;
////////////////	cv::merge(planes, 2, complexImg);
////////////////	cv::dft(complexImg, complexImg);
////////////////	cv::split(complexImg, planes);
////////////////}
////////////////
////////////////void get_myphase_saliency2(Mat input, Mat &output)
////////////////{
////////////////	Mat myphase_in = input.clone();
////////////////
////////////////	MyGammaCorrection(myphase_in, myphase_in, 5);
////////////////	Mat grad_x, grad_y;
////////////////	Mat abs_grad_x, abs_grad_y;
////////////////	Sobel(myphase_in, grad_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
////////////////	convertScaleAbs(grad_x, abs_grad_x);
////////////////	Sobel(myphase_in, grad_y, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
////////////////	convertScaleAbs(grad_y, abs_grad_y);
////////////////	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, myphase_in);
////////////////
////////////////	Mat chushi;
////////////////	get_phase_map(myphase_in, chushi);
////////////////	imshow("chushi", chushi);
////////////////
////////////////
////////////////	//Mat chushi_clone1 = chushi.clone();
////////////////	//cv::resize(chushi_clone1, chushi_clone1, cv::Size(chushi_clone1.cols/10, chushi_clone1.rows/10), cv::INTER_AREA);
////////////////	//Mat low_rank_part, sparse_part;
////////////////	//low_rank_decomposition(chushi_clone1, low_rank_part, sparse_part);
////////////////	//Mat quanju = sparse_part.clone();
////////////////	//cv::resize(quanju, quanju, input.size(), cv::INTER_CUBIC);
////////////////	//imshow("quanju", quanju);
////////////////
////////////////
////////////////
////////////////
////////////////
////////////////
////////////////
////////////////
////////////////
////////////////
////////////////
////////////////
////////////////	//int div = 10;
////////////////	//Mat chushi_clone2 = chushi.clone();
////////////////	//int blocknum = 0;
////////////////	//int blockwidth = chushi_clone2.cols/div;
////////////////	//int blockheight = chushi_clone2.rows/div;
////////////////	//int num = div*div;
////////////////	//std::cout << "chushi.size:"<< chushi_clone2.cols << " " << chushi_clone2.rows<< std::endl;
////////////////	//std::cout << "block.size:" << blockwidth << " " << blockheight << std::endl;
////////////////	//std::cout << "theory blocknum:" << num << std::endl;
////////////////
////////////////	//Mat Y = Mat::zeros(blockheight, num, CV_64FC1);
////////////////	//std::vector<Mat> pk;
////////////////	//for (int y = 0; y < chushi_clone2.cols - blockwidth + 1; y = y + blockwidth)
////////////////	//{
////////////////	//	for (int x = 0; x < chushi_clone2.rows - blockheight + 1; x = x + blockheight)
////////////////	//	{
////////////////	//		cv::Rect rectTmp(y, x, blockwidth, blockheight);
////////////////	//		Mat p = chushi_clone2(rectTmp).clone();
////////////////	//		pk.push_back(p);
////////////////	//		std::vector<double> col;
////////////////	//		for (int i = 0; i < p.rows; i++)
////////////////	//		{
////////////////	//			std::vector<int> row;
////////////////	//			for (int j = 0; j < p.cols; j++)
////////////////	//			{
////////////////	//				int tmp = p.at<int>(i, j);
////////////////	//				row.push_back(tmp);
////////////////	//			}
////////////////	//			int sum = std::accumulate(std::begin(row), std::end(row), 0);
////////////////	//			double mean = (double)sum / row.size(); 
////////////////	//			col.push_back(mean);
////////////////	//		}
////////////////	//		for (int j = 0; j < col.size(); j++)
////////////////	//		{
////////////////	//			Y.at<double>(j, blocknum) = col[j];
////////////////	//		}
////////////////	//		blocknum++;
////////////////	//	}
////////////////	//}
////////////////	//std::cout << "acutally blocknum:" << blocknum << std::endl;
////////////////	//std::cout << "Y.size:" << Y.cols << " " << Y.rows << std::endl;
////////////////	//std::cout << "pk.num:" << pk.size() << std::endl;
////////////////	//std::cout << "Y.type:" << Y.type() << std::endl;
////////////////
////////////////	////Mat temp;
////////////////	//////int multiple = 2;
////////////////	////if (Y.cols > Y.rows) 
////////////////	////{
////////////////	////	int multiple = Y.cols / Y.rows;
////////////////	////	cv::resize(Y, temp, cv::Size(Y.cols / multiple, Y.rows), cv::INTER_AREA);
////////////////	////}
////////////////	////else if (Y.cols <= Y.rows)
////////////////	////{
////////////////	////	int multiple = Y.rows / Y.cols;
////////////////	////	cv::resize(Y, temp, cv::Size(Y.cols, Y.rows / multiple), cv::INTER_AREA);
////////////////	////}
////////////////	//Mat temp(Y);
////////////////	//std::cout << "temp.size:" << temp.cols << " " << temp.rows << std::endl;
////////////////	//Mat Y_sparse_lowrank, Y_sparse;
////////////////	//low_rank_decomposition(temp, Y_sparse_lowrank, Y_sparse);
////////////////	//cv::resize(temp, Y_sparse, Y.size(), cv::INTER_CUBIC);
////////////////	//std::cout << "Y_sparse.size:" << Y_sparse.cols << " " << Y_sparse.rows << std::endl;
////////////////	//
////////////////
////////////////
////////////////	//std::vector<double> S;
////////////////	//for (int k = 0; k < Y_sparse.cols; k++)
////////////////	//{
////////////////	//	Mat col_tmp = Mat::zeros(blockheight, 1, CV_64FC1);
////////////////	//	for (int t = 0; t < col_tmp.rows; t++)
////////////////	//	{
////////////////	//		col_tmp.at<double>(t, 0) = (double)Y_sparse.at<double>(t, k);
////////////////	//	}
////////////////	//	double sp = cv::norm(col_tmp);
////////////////	//	S.push_back(sp);
////////////////	//}
////////////////	//std::cout << "S.size:" << S.size() << std::endl;
////////////////
////////////////
////////////////	//float d1 = 0.8, d2 = 0.2;
////////////////	//float u1 = 0.8, u2 = 0.2;
////////////////	//int block = 0;
////////////////	//for (int y = 0; y < chushi_clone2.cols - blockwidth + 1; y = y + blockwidth)
////////////////	//{
////////////////	//	for (int x = 0; x < chushi_clone2.rows - blockheight + 1; x = x + blockheight)
////////////////	//	{
////////////////	//		cv::Rect rectTmp(y, x, blockwidth, blockheight);
////////////////	//		//Mat p = chushi_clone2(rectTmp);
////////////////	//		if (S[block] > d2 && S[block] <= d1)
////////////////	//		{
////////////////	//			chushi_clone2(rectTmp) = u1*pk[block];
////////////////	//		}
////////////////	//		else if (S[block] <= d2)
////////////////	//		{
////////////////	//			chushi_clone2(rectTmp) = u2 * pk[block];
////////////////	//		}
////////////////	//		else
////////////////	//		{
////////////////	//			continue;
////////////////	//		}
////////////////	//		block++;
////////////////	//	}
////////////////	//}
////////////////	//Mat jubu = chushi_clone2;
////////////////	////imshow("jubu", jubu);
////////////////
////////////////	//float r1 = 0.6, r2 = 0.4;
////////////////	//Mat final = quanju * 0.6 + jubu * 0.4;
////////////////	//cv::normalize(final, final, 0, 255, cv::NORM_MINMAX);
////////////////	//final.convertTo(final, CV_8UC1);
////////////////	////imshow("final", final);
////////////////
////////////////	output = chushi.clone();
////////////////}
////////////////
//////////////////将幅度归一，相角保持不变
////////////////void one_amplitude(Mat &complex_r, Mat &complex_i, Mat &dst)
////////////////{
////////////////	Mat temp[] = { Mat::zeros(complex_r.size(),CV_32FC1), Mat::zeros(complex_r.size(),CV_32FC1) };
////////////////	float realv = 0.0, imaginv = 0.0;
////////////////	for (int i = 0; i < complex_r.cols; i++) {
////////////////		for (int j = 0; j < complex_r.rows; j++) {
////////////////			realv = complex_r.at<float>(i, j);
////////////////			imaginv = complex_i.at<float>(i, j);
////////////////			float distance = sqrt(realv*realv + imaginv * imaginv);
////////////////			temp[0].at<float>(i, j) = realv / distance;
////////////////			temp[1].at<float>(i, j) = imaginv / distance;
////////////////		}
////////////////	}
////////////////	merge(temp, 2, dst);
////////////////}
////////////////
//////////////////将相角归一，幅值保持不变
////////////////void one_angel(Mat &complex_r, Mat &complex_i, Mat &dst)
////////////////{
////////////////	Mat temp[] = { Mat::zeros(complex_r.size(),CV_32FC1), Mat::zeros(complex_r.size(),CV_32FC1) };
////////////////	float realv = 0.0, imaginv = 0.0;
////////////////	for (int i = 0; i < complex_r.cols; i++) {
////////////////		for (int j = 0; j < complex_r.rows; j++) {
////////////////			realv = complex_r.at<float>(i, j);
////////////////			imaginv = complex_i.at<float>(i, j);
////////////////			float distance = sqrt(realv*realv + imaginv * imaginv);
////////////////			temp[0].at<float>(i, j) = distance / sqrt(2);
////////////////			temp[1].at<float>(i, j) = distance / sqrt(2);
////////////////		}
////////////////	}
////////////////	merge(temp, 2, dst);
////////////////}
////////////////
//////////////////使用1的幅值和2的相位合并
////////////////void mixed_amplitude_with_phase(Mat &real1, Mat &imag1, Mat &real2, Mat &imag2, Mat &dst)
////////////////{
////////////////	if (real1.size() != real2.size()) {
////////////////		std::cerr << "image don't ==" << std::endl;
////////////////		return;
////////////////	}
////////////////	Mat temp[] = { Mat::zeros(real1.size(),CV_32FC1), Mat::zeros(real1.size(),CV_32FC1) };
////////////////	float realv1 = 0.0, imaginv1 = 0.0, realv2 = 0.0, imaginv2 = 0.0;
////////////////	for (int i = 0; i < real1.cols; i++) {
////////////////		for (int j = 0; j < real1.rows; j++) {
////////////////			realv1 = real1.at<float>(i, j);
////////////////			imaginv1 = imag1.at<float>(i, j);
////////////////			realv2 = real2.at<float>(i, j);
////////////////			imaginv2 = imag2.at<float>(i, j);
////////////////			float distance1 = sqrt(realv1*realv1 + imaginv1 * imaginv1);
////////////////			float distance2 = sqrt(realv2*realv2 + imaginv2 * imaginv2);
////////////////			temp[0].at<float>(i, j) = (realv2*distance1) / distance2;
////////////////			temp[1].at<float>(i, j) = (imaginv2*distance1) / distance2;
////////////////		}
////////////////	}
////////////////	merge(temp, 2, dst);
////////////////}
////////////////
//////////////////使用1的相位和2的幅值合并
////////////////void mixed_phase_with_amplitude(Mat &real1, Mat &imag1, Mat &real2, Mat &imag2, Mat &dst)
////////////////{
////////////////	if (real1.size() != real2.size()) {
////////////////		std::cerr << "image don't ==" << std::endl;
////////////////		return;
////////////////	}
////////////////	Mat temp[] = { Mat::zeros(real1.size(),CV_32FC1), Mat::zeros(real1.size(),CV_32FC1) };
////////////////	float realv1 = 0.0, imaginv1 = 0.0, realv2 = 0.0, imaginv2 = 0.0;
////////////////	for (int i = 0; i < real1.cols; i++) {
////////////////		for (int j = 0; j < real1.rows; j++) {
////////////////			realv1 = real1.at<float>(i, j);
////////////////			imaginv1 = imag1.at<float>(i, j);
////////////////			realv2 = real2.at<float>(i, j);
////////////////			imaginv2 = imag2.at<float>(i, j);
////////////////			float distance1 = sqrt(realv1*realv1 + imaginv1 * imaginv1);
////////////////			float distance2 = sqrt(realv2*realv2 + imaginv2 * imaginv2);
////////////////			temp[0].at<float>(i, j) = (realv1*distance2) / distance1;
////////////////			temp[1].at<float>(i, j) = (imaginv1*distance2) / distance1;
////////////////		}
////////////////	}
////////////////	merge(temp, 2, dst);
////////////////}
////////////////
////////////////cv::Mat fourior_inverser(Mat &_complexim)
////////////////{
////////////////	Mat dst;
////////////////	Mat iDft[] = { Mat::zeros(_complexim.size(),CV_32F),Mat::zeros(_complexim.size(),CV_32F) };//创建两个通道，类型为float，大小为填充后的尺寸
////////////////	idft(_complexim, _complexim);//傅立叶逆变换
////////////////	split(_complexim, iDft);//结果貌似也是复数
////////////////	magnitude(iDft[0], iDft[1], dst);//分离通道，主要获取0通道
//////////////////    dst += Scalar::all(1);                    // switch to logarithmic scale
//////////////////    log(dst, dst);
////////////////	//归一化处理，float类型的显示范围为0-255,255为白色，0为黑色
////////////////	normalize(dst, dst, 0, 255, cv::NORM_MINMAX);
////////////////	dst.convertTo(dst, CV_8U);
////////////////	return dst;
////////////////}
////////////////
////////////////void move_to_center(Mat &center_img)
////////////////{
////////////////	int cx = center_img.cols / 2;
////////////////	int cy = center_img.rows / 2;
////////////////	Mat q0(center_img, Rect(0, 0, cx, cy));
////////////////	Mat q1(center_img, Rect(cx, 0, cx, cy)); 
////////////////	Mat q2(center_img, Rect(0, cy, cx, cy)); 
////////////////	Mat q3(center_img, Rect(cx, cy, cx, cy));
////////////////
////////////////	Mat tmp;
////////////////	q0.copyTo(tmp);
////////////////	q3.copyTo(q0);
////////////////	tmp.copyTo(q3);
////////////////
////////////////	q1.copyTo(tmp);
////////////////	q2.copyTo(q1);
////////////////	tmp.copyTo(q2);
////////////////}
////////////////
////////////////void fast_dft(cv::Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img)
////////////////{
////////////////	src_img.convertTo(src_img, CV_32FC1);
////////////////
////////////////	
////////////////	int oph = cv::getOptimalDFTSize(src_img.rows);
////////////////	int opw = cv::getOptimalDFTSize(src_img.cols);
////////////////	Mat padded;
////////////////	copyMakeBorder(src_img, padded, 0, oph - src_img.rows, 0, opw - src_img.cols,
////////////////		cv::BORDER_CONSTANT, cv::Scalar::all(0));
////////////////
////////////////	Mat temp[] = { padded, Mat::zeros(padded.size(),CV_32FC1) };
////////////////	Mat complexI;
////////////////	merge(temp, 2, complexI);
////////////////	dft(complexI, complexI);
////////////////	split(complexI, temp);
////////////////	temp[0].copyTo(real_img);
////////////////	temp[1].copyTo(ima_img);
////////////////}
//////////////
//////////////int main(int argc, char *argv[])
//////////////{
//////////////	//	Mat image = cv::imread("1.jpg", 0);
//////////////	//	Mat image2 = cv::imread("2.jpg", 0);
//////////////	//	if (image.empty() || image2.empty())
//////////////	//		return -1;
//////////////	//	imshow("woman_src", image);
//////////////	//	//resize(image2, image2, cv::Size(), 0.5, 0.5);
//////////////	//	imshow("sqrt_src", image2);
//////////////	//
//////////////	//	Mat woman_real, woman_imag;
//////////////	//	Mat sqrt_real, sqrt_imag;
//////////////	//
//////////////	//	fast_dft(image, woman_real, woman_imag);
//////////////	//	fast_dft(image2, sqrt_real, sqrt_imag);
//////////////	//
//////////////	//	Mat img_range, img_angle;
//////////////	//	one_amplitude(woman_real, woman_imag, img_range); 
//////////////	//	one_angel(woman_real, woman_imag, img_angle);
//////////////	//
//////////////	//	Mat woman_amp2sqrt_angle, sqrt_amp2woman_angle;
//////////////	//	mixed_amplitude_with_phase(woman_real, woman_imag,
//////////////	//		sqrt_real, sqrt_imag, woman_amp2sqrt_angle);
//////////////	//	mixed_phase_with_amplitude(woman_real, woman_imag,
//////////////	//		sqrt_real, sqrt_imag, sqrt_amp2woman_angle);
//////////////	//	Mat amplitude, angle, amplitude_src;
//////////////	//	magnitude(woman_real, woman_imag, amplitude);
//////////////	//	phase(woman_real, woman_imag, angle); 
//////////////	////    cartToPolar(temp[0], temp[1],amplitude, angle);
//////////////	//
//////////////	//	move_to_center(amplitude);
//////////////	//
//////////////	//	divide(amplitude, amplitude.cols*amplitude.rows, amplitude_src);
//////////////	//	imshow("amplitude_src", amplitude_src);
//////////////	//
//////////////	//	amplitude += cv::Scalar::all(1);
//////////////	//	log(amplitude, amplitude);
//////////////	//	normalize(amplitude, amplitude, 0, 255, cv::NORM_MINMAX);
//////////////	//	amplitude.convertTo(amplitude, CV_8U);
//////////////	//	imshow("amplitude", amplitude);
//////////////	//
//////////////	//	normalize(angle, angle, 0, 255, cv::NORM_MINMAX);
//////////////	//	angle.convertTo(angle, CV_8U);
//////////////	//	imshow("angle", angle);
//////////////	//
//////////////	//	//*******************************************************************
//////////////	//
//////////////	//	Mat inverse_amp = fourior_inverser(img_range); 
//////////////	//	Mat inverse_angle = fourior_inverser(img_angle);
//////////////	//	Mat inverse_dst1 = fourior_inverser(woman_amp2sqrt_angle);
//////////////	//	Mat inverse_dst2 = fourior_inverser(sqrt_amp2woman_angle);
//////////////	//	move_to_center(inverse_angle);
//////////////	//	imshow("inverse_angle", inverse_amp);
//////////////	//	imshow("inverse_amp", inverse_angle);
//////////////	//	imshow("woman_amp2sqrt_angle", inverse_dst1);
//////////////	//	imshow("sqrt_amp2woman_angle", inverse_dst2);
//////////////
//////////////		//imwrite("phase.jpg", angle);
//////////////
//////////////	for (int i = 1; i < 100; i++)
//////////////	{
//////////////		Mat in = cv::imread(std::to_string(i) + ".jpg");
//////////////		if (in.empty())
//////////////			continue;
//////////////		Mat myphase2 = in.clone();
//////////////		vector<cv::Rect> myphase2_rects;
//////////////		get_saliency_rects(myphase2, myphase2_type_, myphase2_rects);
//////////////		show_rects("myphase2", myphase2, myphase2_rects);
//////////////
//////////////
//////////////	}
//////////////
//////////////	cv::waitKey(0);
//////////////	return 0;
//////////////}
