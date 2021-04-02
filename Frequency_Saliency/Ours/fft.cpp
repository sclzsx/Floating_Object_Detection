#include "fft.h"


//将幅度归一，相角保持不变
void one_amplitude(Mat complex_r, Mat complex_i, Mat &dst)
{
	Mat temp[] = { Mat::zeros(complex_r.size(),CV_64FC1), Mat::zeros(complex_i.size(),CV_64FC1) };
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

void fast_dft(cv::Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img)
{
	Mat sr_in = src_img.clone();
	if (sr_in.channels() == 3)
		cvtColor(sr_in, sr_in, cv::COLOR_RGB2GRAY);
	Mat planes[] = { cv::Mat_<float>(sr_in), Mat::zeros(sr_in.size(), CV_64FC1) };
	Mat complexI; //复数矩阵
	merge(planes, 2, complexI); //把单通道矩阵组合成复数形式的双通道矩阵
	dft(complexI, complexI);  // 使用离散傅立叶变换
	split(complexI, planes); //分离复数到实部和虚部
	real_img = planes[0]; //实部
	ima_img = planes[1]; //虚部


	//if (src_img.channels() != 1)
	//	cvtColor(src_img, src_img, cv::COLOR_RGB2GRAY);
	//src_img.convertTo(src_img, CV_64FC1);
	/////////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
	//int oph = cv::getOptimalDFTSize(src_img.rows);
	//int opw = cv::getOptimalDFTSize(src_img.cols);
	//Mat padded;
	//copyMakeBorder(src_img, padded, 0, oph - src_img.rows, 0, opw - src_img.cols,
	//	cv::BORDER_CONSTANT, cv::Scalar::all(0));
	//Mat temp[] = { padded, Mat::zeros(padded.size(),CV_64FC1) };
	//Mat complexI;
	//merge(temp, 2, complexI);
	//dft(complexI, complexI);    //傅里叶变换
	//split(complexI, temp);      //显示频谱图
	//temp[0].copyTo(real_img);
	//temp[1].copyTo(ima_img);
}
