#ifndef FFT_H_
#define FFT_H_

#include <opencv2/opencv.hpp>

using cv::Mat;

//�����ȹ�һ����Ǳ��ֲ���
void one_amplitude(Mat complex_r, Mat complex_i, Mat &dst);

cv::Mat fourior_inverser(Mat &_complexim);

void fast_dft(cv::Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img);

#endif // !FFT_H_

