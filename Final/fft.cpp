#include "fft.h"

void fast_dft(cv::Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img)
{
	if (src_img.channels() != 1)
		cvtColor(src_img, src_img, cv::COLOR_RGB2GRAY);

	src_img.convertTo(src_img, CV_64FC1);

	///////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
	int oph = cv::getOptimalDFTSize(src_img.rows);
	int opw = cv::getOptimalDFTSize(src_img.cols);
	Mat padded;
	copyMakeBorder(src_img, padded, 0, oph - src_img.rows, 0, opw - src_img.cols,
		cv::BORDER_CONSTANT, cv::Scalar::all(0));

	Mat temp[] = { padded, Mat::zeros(padded.size(),CV_64FC1) };
	Mat complexI;
	merge(temp, 2, complexI);
	dft(complexI, complexI);    //傅里叶变换
	split(complexI, temp);      //显示频谱图
	temp[0].copyTo(real_img);
	temp[1].copyTo(ima_img);
}

//将幅度归一，相角保持不变
void one_amplitude(Mat &complex_r, Mat &complex_i, Mat &dst)
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

//将相角归一，幅值保持不变
void one_angel(Mat &complex_r, Mat &complex_i, Mat &dst)
{
	Mat temp[] = { Mat::zeros(complex_r.size(),CV_64FC1), Mat::zeros(complex_r.size(),CV_64FC1) };
	float realv = 0.0, imaginv = 0.0;
	for (int i = 0; i < complex_r.cols; i++) {
		for (int j = 0; j < complex_r.rows; j++) {
			realv = complex_r.at<float>(i, j);
			imaginv = complex_i.at<float>(i, j);
			float distance = sqrt(realv*realv + imaginv * imaginv);
			temp[0].at<float>(i, j) = distance / sqrt(2);
			temp[1].at<float>(i, j) = distance / sqrt(2);
		}
	}
	merge(temp, 2, dst);
}

//使用1的幅值和2的相位合并
void mixed_amplitude_with_phase(Mat &real1, Mat &imag1, Mat &real2, Mat &imag2, Mat &dst)
{
	if (real1.size() != real2.size()) {
		std::cerr << "image don't ==" << std::endl;
		return;
	}
	Mat temp[] = { Mat::zeros(real1.size(),CV_64FC1), Mat::zeros(real1.size(),CV_64FC1) };
	float realv1 = 0.0, imaginv1 = 0.0, realv2 = 0.0, imaginv2 = 0.0;
	for (int i = 0; i < real1.cols; i++) {
		for (int j = 0; j < real1.rows; j++) {
			realv1 = real1.at<float>(i, j);
			imaginv1 = imag1.at<float>(i, j);
			realv2 = real2.at<float>(i, j);
			imaginv2 = imag2.at<float>(i, j);
			float distance1 = sqrt(realv1*realv1 + imaginv1 * imaginv1);
			float distance2 = sqrt(realv2*realv2 + imaginv2 * imaginv2);
			temp[0].at<float>(i, j) = (realv2*distance1) / distance2;
			temp[1].at<float>(i, j) = (imaginv2*distance1) / distance2;
		}
	}
	merge(temp, 2, dst);
}

//使用1的相位和2的幅值合并
void mixed_phase_with_amplitude(Mat &real1, Mat &imag1, Mat &real2, Mat &imag2, Mat &dst)
{
	if (real1.size() != real2.size()) {
		std::cerr << "image don't ==" << std::endl;
		return;
	}
	Mat temp[] = { Mat::zeros(real1.size(),CV_64FC1), Mat::zeros(real1.size(),CV_64FC1) };
	float realv1 = 0.0, imaginv1 = 0.0, realv2 = 0.0, imaginv2 = 0.0;
	for (int i = 0; i < real1.cols; i++) {
		for (int j = 0; j < real1.rows; j++) {
			realv1 = real1.at<float>(i, j);
			imaginv1 = imag1.at<float>(i, j);
			realv2 = real2.at<float>(i, j);
			imaginv2 = imag2.at<float>(i, j);
			float distance1 = sqrt(realv1*realv1 + imaginv1 * imaginv1);
			float distance2 = sqrt(realv2*realv2 + imaginv2 * imaginv2);
			temp[0].at<float>(i, j) = (realv1*distance2) / distance1;
			temp[1].at<float>(i, j) = (imaginv1*distance2) / distance1;
		}
	}
	merge(temp, 2, dst);
}

//cv::Mat fourior_inverser(Mat &_complexim)
//{
//	Mat dst;
//	Mat iDft[] = { Mat::zeros(_complexim.size(),CV_32F),Mat::zeros(_complexim.size(),CV_32F) };//创建两个通道，类型为float，大小为填充后的尺寸
//	idft(_complexim, _complexim);//傅立叶逆变换
//	split(_complexim, iDft);//结果貌似也是复数
//	magnitude(iDft[0], iDft[1], dst);//分离通道，主要获取0通道
////    dst += Scalar::all(1);                    // switch to logarithmic scale
////    log(dst, dst);
//	//归一化处理，float类型的显示范围为0-255,255为白色，0为黑色
//	normalize(dst, dst, 0, 255, cv::NORM_MINMAX);
//	dst.convertTo(dst, CV_8U);
//	return dst;
//}




void move_to_center(Mat &center_img)
{
	int cx = center_img.cols / 2;
	int cy = center_img.rows / 2;
	Mat q0(center_img, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(center_img, cv::Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(center_img, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(center_img, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

