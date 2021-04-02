//#include "opencv2/opencv.hpp"
//
//using namespace cv;
//
////将幅度归一，相角保持不变
//void one_amplitude(Mat &complex_r, Mat &complex_i, Mat &dst)
//{
//	Mat temp[] = { Mat::zeros(complex_r.size(),CV_32FC1), Mat::zeros(complex_r.size(),CV_32FC1) };
//	float realv = 0.0, imaginv = 0.0;
//	for (int i = 0; i < complex_r.cols; i++) {
//		for (int j = 0; j < complex_r.rows; j++) {
//			realv = complex_r.at<float>(i, j);
//			imaginv = complex_i.at<float>(i, j);
//			float distance = sqrt(realv*realv + imaginv * imaginv);
//			temp[0].at<float>(i, j) = realv / distance;
//			temp[1].at<float>(i, j) = imaginv / distance;
//		}
//	}
//	merge(temp, 2, dst);
//}
//
////将相角归一，幅值保持不变
//void one_angel(Mat &complex_r, Mat &complex_i, Mat &dst)
//{
//	Mat temp[] = { Mat::zeros(complex_r.size(),CV_32FC1), Mat::zeros(complex_r.size(),CV_32FC1) };
//	float realv = 0.0, imaginv = 0.0;
//	for (int i = 0; i < complex_r.cols; i++) {
//		for (int j = 0; j < complex_r.rows; j++) {
//			realv = complex_r.at<float>(i, j);
//			imaginv = complex_i.at<float>(i, j);
//			float distance = sqrt(realv*realv + imaginv * imaginv);
//			temp[0].at<float>(i, j) = distance / sqrt(2);
//			temp[1].at<float>(i, j) = distance / sqrt(2);
//		}
//	}
//	merge(temp, 2, dst);
//}
//
////使用1的幅值和2的相位合并
//void mixed_amplitude_with_phase(Mat &real1, Mat &imag1, Mat &real2, Mat &imag2, Mat &dst)
//{
//	if (real1.size() != real2.size()) {
//		std::cerr << "image don't ==" << std::endl;
//		return;
//	}
//	Mat temp[] = { Mat::zeros(real1.size(),CV_32FC1), Mat::zeros(real1.size(),CV_32FC1) };
//	float realv1 = 0.0, imaginv1 = 0.0, realv2 = 0.0, imaginv2 = 0.0;
//	for (int i = 0; i < real1.cols; i++) {
//		for (int j = 0; j < real1.rows; j++) {
//			realv1 = real1.at<float>(i, j);
//			imaginv1 = imag1.at<float>(i, j);
//			realv2 = real2.at<float>(i, j);
//			imaginv2 = imag2.at<float>(i, j);
//			float distance1 = sqrt(realv1*realv1 + imaginv1 * imaginv1);
//			float distance2 = sqrt(realv2*realv2 + imaginv2 * imaginv2);
//			temp[0].at<float>(i, j) = (realv2*distance1) / distance2;
//			temp[1].at<float>(i, j) = (imaginv2*distance1) / distance2;
//		}
//	}
//	merge(temp, 2, dst);
//}
//
////使用1的相位和2的幅值合并
//void mixed_phase_with_amplitude(Mat &real1, Mat &imag1, Mat &real2, Mat &imag2, Mat &dst)
//{
//	if (real1.size() != real2.size()) {
//		std::cerr << "image don't ==" << std::endl;
//		return;
//	}
//	Mat temp[] = { Mat::zeros(real1.size(),CV_32FC1), Mat::zeros(real1.size(),CV_32FC1) };
//	float realv1 = 0.0, imaginv1 = 0.0, realv2 = 0.0, imaginv2 = 0.0;
//	for (int i = 0; i < real1.cols; i++) {
//		for (int j = 0; j < real1.rows; j++) {
//			realv1 = real1.at<float>(i, j);
//			imaginv1 = imag1.at<float>(i, j);
//			realv2 = real2.at<float>(i, j);
//			imaginv2 = imag2.at<float>(i, j);
//			float distance1 = sqrt(realv1*realv1 + imaginv1 * imaginv1);
//			float distance2 = sqrt(realv2*realv2 + imaginv2 * imaginv2);
//			temp[0].at<float>(i, j) = (realv1*distance2) / distance1;
//			temp[1].at<float>(i, j) = (imaginv1*distance2) / distance1;
//		}
//	}
//	merge(temp, 2, dst);
//}
//
//cv::Mat fourior_inverser(Mat &_complexim)
//{
//	Mat dst;
//	Mat iDft[] = { Mat::zeros(_complexim.size(),CV_32F),Mat::zeros(_complexim.size(),CV_32F) };
//	idft(_complexim, _complexim);
//	split(_complexim, iDft);
//	magnitude(iDft[0], iDft[1], dst);
////    dst += Scalar::all(1);                    // switch to logarithmic scale
////    log(dst, dst);
//	//归一化处理，float类型的显示范围为0-255,255为白色，0为黑色
//	normalize(dst, dst, 0, 255, NORM_MINMAX);
//	dst.convertTo(dst, CV_8U);
//	return dst;
//}
//
//void move_to_center(Mat &center_img)
//{
//	int cx = center_img.cols / 2;
//	int cy = center_img.rows / 2;
//	Mat q0(center_img, Rect(0, 0, cx, cy)); 
//	Mat q1(center_img, Rect(cx, 0, cx, cy)); 
//	Mat q2(center_img, Rect(0, cy, cx, cy));
//	Mat q3(center_img, Rect(cx, cy, cx, cy)); 
//
//	Mat tmp; 
//	q0.copyTo(tmp);
//	q3.copyTo(q0);
//	tmp.copyTo(q3);
//
//	q1.copyTo(tmp);
//	q2.copyTo(q1);
//	tmp.copyTo(q2);
//}
//
//void fast_dft(cv::Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img)
//{
//	src_img.convertTo(src_img, CV_32FC1);
//
//	///////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
//	int oph = getOptimalDFTSize(src_img.rows);
//	int opw = getOptimalDFTSize(src_img.cols);
//	Mat padded;
//	copyMakeBorder(src_img, padded, 0, oph - src_img.rows, 0, opw - src_img.cols,
//		BORDER_CONSTANT, Scalar::all(0));
//
//	Mat temp[] = { padded, Mat::zeros(padded.size(),CV_32FC1) };
//	Mat complexI;
//	merge(temp, 2, complexI);
//	dft(complexI, complexI); 
//	split(complexI, temp); 
//	temp[0].copyTo(real_img);
//	temp[1].copyTo(ima_img);
//}
//
//int main(int argc, char *argv[])
//{
//	Mat image = imread("bridge.jpg", IMREAD_GRAYSCALE);
//	Mat image2 = imread("leaf.jpg", IMREAD_GRAYSCALE);
//	//if (image.empty() || image2.empty())
//	//	return -1;
//	imshow("woman_src", image);
//	//resize(image2, image2, cv::Size(), 0.5, 0.5);
//	imshow("sqrt_src", image2);
//
//	Mat woman_real, woman_imag;
//	Mat sqrt_real, sqrt_imag;
//
//	fast_dft(image, woman_real, woman_imag);
//	fast_dft(image2, sqrt_real, sqrt_imag);
//
//	Mat img_range, img_angle;
//	one_amplitude(woman_real, woman_imag, img_range); 
//	one_angel(woman_real, woman_imag, img_angle); 
//
//	Mat woman_amp2sqrt_angle, sqrt_amp2woman_angle;
//	mixed_amplitude_with_phase(woman_real, woman_imag,
//		sqrt_real, sqrt_imag, woman_amp2sqrt_angle);
//	mixed_phase_with_amplitude(woman_real, woman_imag,
//		sqrt_real, sqrt_imag, sqrt_amp2woman_angle);
//	Mat amplitude, angle, amplitude_src;
//	magnitude(woman_real, woman_imag, amplitude);
//	phase(woman_real, woman_imag, angle); 
////    cartToPolar(temp[0], temp[1],amplitude, angle);
//
//	move_to_center(amplitude); 
//
//	divide(amplitude, amplitude.cols*amplitude.rows, amplitude_src);
//	imshow("amplitude_src", amplitude_src);
//
//	amplitude += Scalar::all(1);
//	log(amplitude, amplitude);
//	normalize(amplitude, amplitude, 0, 255, NORM_MINMAX); 
//	amplitude.convertTo(amplitude, CV_8U);
//	imshow("amplitude", amplitude);
//
//	normalize(angle, angle, 0, 255, NORM_MINMAX); 
//	angle.convertTo(angle, CV_8U);
//	imshow("angle", angle);
//
//	//*******************************************************************
//
//	Mat inverse_amp = fourior_inverser(img_range); 
//	Mat inverse_angle = fourior_inverser(img_angle);
//	Mat inverse_dst1 = fourior_inverser(woman_amp2sqrt_angle);
//	Mat inverse_dst2 = fourior_inverser(sqrt_amp2woman_angle);
//	move_to_center(inverse_angle);
//	imshow("inverse_angle", inverse_amp);
//	imshow("inverse_amp", inverse_angle);
//	imshow("woman_amp2sqrt_angle", inverse_dst1);
//	imshow("sqrt_amp2woman_angle", inverse_dst2);
//
//	imwrite("phase.jpg", angle);
//	waitKey(0);
//
//	return 1;
//}
