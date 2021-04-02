#ifndef FFT_H_
#define FFT_H_

#include <opencv2/opencv.hpp>

using cv::Mat;

//将幅度归一，相角保持不变
void one_amplitude(Mat &complex_r, Mat &complex_i, Mat &dst);
//将相角归一，幅值保持不变
void one_angel(Mat &complex_r, Mat &complex_i, Mat &dst);
//使用1的幅值和2的相位合并
void mixed_amplitude_with_phase(Mat &real1, Mat &imag1, Mat &real2, Mat &imag2, Mat &dst);

//使用1的相位和2的幅值合并
void mixed_phase_with_amplitude(Mat &real1, Mat &imag1, Mat &real2, Mat &imag2, Mat &dst);

cv::Mat fourior_inverser(Mat &_complexim);
void move_to_center(Mat &center_img);

void fast_dft(cv::Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img);

#endif // !FFT_H_

