#ifndef FFT_H_
#define FFT_H_

#include <opencv2/opencv.hpp>

using cv::Mat;

//�����ȹ�һ����Ǳ��ֲ���
void one_amplitude(Mat &complex_r, Mat &complex_i, Mat &dst);
//����ǹ�һ����ֵ���ֲ���
void one_angel(Mat &complex_r, Mat &complex_i, Mat &dst);
//ʹ��1�ķ�ֵ��2����λ�ϲ�
void mixed_amplitude_with_phase(Mat &real1, Mat &imag1, Mat &real2, Mat &imag2, Mat &dst);

//ʹ��1����λ��2�ķ�ֵ�ϲ�
void mixed_phase_with_amplitude(Mat &real1, Mat &imag1, Mat &real2, Mat &imag2, Mat &dst);

cv::Mat fourior_inverser(Mat &_complexim);
void move_to_center(Mat &center_img);

void fast_dft(cv::Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img);

#endif // !FFT_H_

